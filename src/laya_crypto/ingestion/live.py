"""Public (unauthenticated) Binance USD-M REST endpoints for inference-time updates.

Read-only market data only. No API key, no order endpoints. The archive lags by about a day,
so inference tops the cache up with the most recent closed candles from here.
"""
from __future__ import annotations

import time

import httpx
import pandas as pd

from ..config import DataConfig
from ..utils.io import read_json, write_json
from ..utils.timeframes import tf_ms
from .parsers import KLINE_COLS


class LiveDataError(RuntimeError):
    """Raised when live public data cannot be fetched; callers must not fabricate a fallback."""


def _client(cfg: DataConfig) -> httpx.Client:
    return httpx.Client(base_url=cfg.fapi_base_url, timeout=20,
                        headers={"User-Agent": "laya-crypto-trainer/0.1 (research)"})


def _get(cfg: DataConfig, path: str, params: dict) -> list | dict:
    try:
        with _client(cfg) as c:
            r = c.get(path, params=params)
    except httpx.HTTPError as e:
        raise LiveDataError(f"{cfg.fapi_base_url}{path}: {e!r}") from e
    if r.status_code == 451:
        raise LiveDataError("Binance futures API refused this location (HTTP 451). "
                            "Use cached archive data or run from a permitted network.")
    if r.status_code != 200:
        raise LiveDataError(f"{path}: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


KLINE_ENDPOINTS = {"klines": ("/fapi/v1/klines", "symbol"), "mark_price_klines": ("/fapi/v1/markPriceKlines", "symbol"),
                   "index_price_klines": ("/fapi/v1/indexPriceKlines", "pair"),
                   "premium_index_klines": ("/fapi/v1/premiumIndexKlines", "symbol")}


def fetch_recent_klines(cfg: DataConfig, symbol: str, timeframe: str, limit: int = 1000,
                        kind: str = "klines") -> pd.DataFrame:
    """Most recent CLOSED candles (the still-forming candle is dropped)."""
    path, key = KLINE_ENDPOINTS[kind]
    raw = _get(cfg, path, {key: symbol, "interval": timeframe, "limit": min(limit, 1500)})
    df = pd.DataFrame(raw, columns=KLINE_COLS).drop(columns=["ignore"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    now = int(time.time() * 1000)
    return df[df["open_time"] + tf_ms(timeframe) <= now].reset_index(drop=True)


def fetch_funding_history(cfg: DataConfig, symbol: str, limit: int = 100) -> pd.DataFrame:
    raw = _get(cfg, "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])
    return pd.DataFrame({"funding_time": df["fundingTime"].astype("int64"),
                         "funding_rate": pd.to_numeric(df["fundingRate"])})


def fetch_open_interest_hist(cfg: DataConfig, symbol: str, period: str = "5m", limit: int = 500) -> pd.DataFrame:
    raw = _get(cfg, "/futures/data/openInterestHist", {"symbol": symbol, "period": period, "limit": limit})
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=["time", "sum_open_interest", "sum_open_interest_value"])
    return pd.DataFrame({"time": df["timestamp"].astype("int64"),
                         "sum_open_interest": pd.to_numeric(df["sumOpenInterest"]),
                         "sum_open_interest_value": pd.to_numeric(df["sumOpenInterestValue"])})


def exchange_info_path(cfg: DataConfig):
    return cfg.root_path / "metadata" / "exchange_info.json"


def fetch_exchange_info(cfg: DataConfig) -> dict:
    """Current contract metadata. Saved with its fetch time: it is NOT historical metadata."""
    info = _get(cfg, "/fapi/v1/exchangeInfo", {})
    symbols = {}
    for s in info.get("symbols", []):
        filters = {f["filterType"]: f for f in s.get("filters", [])}
        symbols[s["symbol"]] = {
            "status": s.get("status"), "contract_type": s.get("contractType"), "quote": s.get("quoteAsset"),
            "onboard_date_ms": s.get("onboardDate"), "delivery_date_ms": s.get("deliveryDate"),
            "tick_size": float(filters.get("PRICE_FILTER", {}).get("tickSize", "nan")),
            "step_size": float(filters.get("LOT_SIZE", {}).get("stepSize", "nan")),
            "min_qty": float(filters.get("LOT_SIZE", {}).get("minQty", "nan")),
            "min_notional": float(filters.get("MIN_NOTIONAL", {}).get("notional", "nan")),
        }
    snap = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "current exchange metadata snapshot; not valid for past filter values",
            "symbols": symbols}
    write_json(exchange_info_path(cfg), snap)
    return snap


def load_exchange_info(cfg: DataConfig) -> dict | None:
    return read_json(exchange_info_path(cfg))
