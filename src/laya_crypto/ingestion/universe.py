"""Point-in-time symbol universe without survivorship bias.

The candidate list is every USDT perpetual that ever had a kline archive, delisted contracts
included. For each month M, a symbol is eligible only on information available before M starts:

1. it had traded for at least `min_history_days` daily candles before M;
2. its trailing `rank_window_days` average daily quote volume (ending the day before M) is at
   least `min_avg_daily_quote_volume`;
3. it ranks within the top `top_n` by that trailing volume.

Today's volume ranking is never used to choose the past.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..config import DataConfig
from . import store


def candidate_symbols(all_symbols: list[str], cfg: DataConfig) -> list[str]:
    u = cfg.universe
    out = []
    for s in all_symbols:
        if not s.endswith(u.quote_asset) or "_" in s or not s.isascii() or not s.isalnum():
            continue
        if s in u.exclude_symbols:
            continue
        out.append(s)
    return sorted(set(out) | (set(u.include_symbols) - set(u.exclude_symbols)))


def daily_volume_panel(cfg: DataConfig, symbols: list[str]) -> pd.DataFrame:
    """Wide frame: index = UTC day, columns = symbols, values = daily quote volume (NaN = not trading)."""
    cols = {}
    for s in symbols:
        df = store.load(cfg, "klines", s, "1d")
        if df is None or df.empty:
            continue
        day = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
        cols[s] = pd.Series(df["quote_volume"].to_numpy(), index=day)
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def build_universe(cfg: DataConfig, panel: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Rows (month, symbol, rank, trailing_avg_quote_volume, history_days) for each eligible symbol."""
    u = cfg.universe
    rows = []
    months = pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS", tz="UTC")
    for m in months:
        hist = panel[panel.index < m]  # strictly before the month starts
        if hist.empty:
            continue
        history_days = hist.notna().sum()
        window = hist[hist.index >= m - pd.Timedelta(days=u.rank_window_days)]
        avg_qv = window.mean(skipna=True)
        # a symbol that stopped trading inside the window is not eligible (no data on its last day)
        traded_last_day = hist.iloc[-1].notna()
        ok = (history_days >= u.min_history_days) & (avg_qv >= u.min_avg_daily_quote_volume) & traded_last_day
        ranked = avg_qv[ok].sort_values(ascending=False).head(u.top_n)
        for rank, (sym, qv) in enumerate(ranked.items(), start=1):
            rows.append({"month": m, "symbol": sym, "rank": rank, "trailing_avg_quote_volume": float(qv),
                         "history_days": int(history_days[sym])})
    out = pd.DataFrame(rows)
    if u.max_symbols and not out.empty:
        keep = out.groupby("symbol")["rank"].min().sort_values().head(u.max_symbols).index
        out = out[out["symbol"].isin(keep)]
    return out


def membership_mask(universe: pd.DataFrame, symbol: str, open_time_ms: pd.Series) -> pd.Series:
    """True where the bar's month had `symbol` in the point-in-time universe."""
    months = set(universe.loc[universe["symbol"] == symbol, "month"])
    bar_month = pd.to_datetime(open_time_ms, unit="ms", utc=True).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    return bar_month.isin(months)


def universe_path(cfg: DataConfig):
    return cfg.root_path / "universe" / "universe.parquet"
