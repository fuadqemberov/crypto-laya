"""Parse Binance archive zips into normalised, UTC-millisecond DataFrames.

Older archive files have no header row and newer ones do; both are detected. Timestamps of any
unit (s / ms / us) are normalised to epoch milliseconds. Rows are de-duplicated on their time key
and sorted. Nothing is filled in: a missing period stays missing and is reported by validation.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.timeframes import to_utc_ms

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
METRICS_COLS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
                "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
AGG_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time",
            "is_buyer_maker"]
DEPTH_COLS = ["timestamp", "percentage", "depth", "notional"]


def _read_csv(path: Path, cols: list[str], **kw) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        raw = z.read(name)
    first = raw[:200].split(b"\n", 1)[0].decode("utf-8", "ignore")
    has_header = not first[:1].isdigit() and not first.startswith("-")
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, **kw)
    if not has_header:
        df.columns = cols[: df.shape[1]]
    return df


def parse_klines(path: Path) -> pd.DataFrame:
    df = _read_csv(path, KLINE_COLS)
    df = df[[c for c in KLINE_COLS if c != "ignore" and c in df.columns]]
    df["open_time"] = to_utc_ms(df["open_time"]).to_numpy()
    df["close_time"] = to_utc_ms(df["close_time"]).to_numpy()
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume",
              "taker_buy_quote_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("float64")
    return df


def parse_funding(path: Path) -> pd.DataFrame:
    df = _read_csv(path, FUNDING_COLS)
    out = pd.DataFrame({"funding_time": to_utc_ms(df["calc_time"]).to_numpy(),
                        "funding_interval_hours": pd.to_numeric(df["funding_interval_hours"], errors="coerce"),
                        "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce")})
    return out


def parse_metrics(path: Path) -> pd.DataFrame:
    df = _read_csv(path, METRICS_COLS)
    ts = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    out = pd.DataFrame({"time": (ts.astype("int64") // 1_000_000).to_numpy()})
    for c in METRICS_COLS[2:]:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return out[ts.notna().to_numpy()]


def parse_agg_trades_to_flow(path: Path, bar_ms: int = 60_000) -> pd.DataFrame:
    """Aggregate trades to order-flow bars: aggressive buy/sell volume and trade counts.

    `is_buyer_maker == True` means the seller was the aggressor (taker sell).
    """
    df = _read_csv(path, AGG_COLS, usecols=None)
    t = to_utc_ms(df["transact_time"]).to_numpy()
    qty = pd.to_numeric(df["quantity"], errors="coerce").to_numpy()
    px = pd.to_numeric(df["price"], errors="coerce").to_numpy()
    maker = df["is_buyer_maker"].astype(str).str.lower().eq("true").to_numpy()
    bar = (t // bar_ms) * bar_ms
    g = pd.DataFrame({"open_time": bar, "buy_qty": np.where(maker, 0.0, qty), "sell_qty": np.where(maker, qty, 0.0),
                      "buy_quote": np.where(maker, 0.0, qty * px), "sell_quote": np.where(maker, qty * px, 0.0),
                      "trades": 1})
    return g.groupby("open_time", as_index=False).sum()


def parse_book_depth(path: Path) -> pd.DataFrame:
    """Depth snapshots -> one row per snapshot with bid/ask notional within 1% and 5% of mid."""
    df = _read_csv(path, DEPTH_COLS)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(time=(ts.astype("int64") // 1_000_000), pct=pd.to_numeric(df["percentage"], errors="coerce"),
                   notional=pd.to_numeric(df["notional"], errors="coerce"))[ts.notna().to_numpy()]
    piv = df.pivot_table(index="time", columns="pct", values="notional", aggfunc="last")
    out = pd.DataFrame(index=piv.index)
    for lvl in (1, 5):
        out[f"bid_notional_{lvl}pct"] = piv.get(-lvl)
        out[f"ask_notional_{lvl}pct"] = piv.get(lvl)
    return out.reset_index()


PARSERS = {
    "klines": parse_klines,
    "mark_price_klines": parse_klines,
    "index_price_klines": parse_klines,
    "premium_index_klines": parse_klines,
    "funding_rate": parse_funding,
    "metrics": parse_metrics,
    "agg_trades": parse_agg_trades_to_flow,
    "book_depth": parse_book_depth,
}
TIME_KEY = {
    "klines": "open_time", "mark_price_klines": "open_time", "index_price_klines": "open_time",
    "premium_index_klines": "open_time", "funding_rate": "funding_time", "metrics": "time",
    "agg_trades": "open_time", "book_depth": "time",
}


def normalise(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, int]:
    """Sort by the time key and drop duplicate timestamps (keeping the last). Returns (df, dropped)."""
    key = TIME_KEY[kind]
    n = len(df)
    if kind == "agg_trades":  # a bar split across two files is summed, not de-duplicated
        df = df.groupby(key, as_index=False).sum()
    df = df.dropna(subset=[key]).sort_values(key, kind="stable")
    df = df.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
    df[key] = df[key].astype("int64")
    return df, n - len(df)
