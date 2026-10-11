"""Causal technical indicators on pandas Series.

Every function uses only the current and past rows (rolling windows, or `ewm(adjust=False)`,
which is recursive), so the value at bar t is known at the close of bar t. Wilder smoothing
(alpha = 1/n) is used where the original definitions use it (RSI, ATR, ADX).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    down = wilder((-d).clip(lower=0), n)
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(down != 0, 100.0).where(up.notna())
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = wilder(true_range(high, low, close), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def roc(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1.0


def momentum(close: pd.Series, n: int) -> pd.Series:
    return close - close.shift(n)


def stoch_rsi(close: pd.Series, rsi_n: int = 14, stoch_n: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    r = rsi(close, rsi_n)
    lo = r.rolling(stoch_n, min_periods=stoch_n).min()
    hi = r.rolling(stoch_n, min_periods=stoch_n).max()
    raw = (r - lo) / (hi - lo).replace(0, np.nan)
    kk = raw.rolling(k, min_periods=k).mean()
    return pd.DataFrame({"stoch_rsi_k": kk, "stoch_rsi_d": kk.rolling(d, min_periods=d).mean()})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame({"bb_width": (upper - lower) / mid, "bb_pctb": (close - lower) / (upper - lower)})


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def realized_vol(close: pd.Series, n: int) -> pd.Series:
    """Standard deviation of log returns over the last n bars (not annualised)."""
    return log_returns(close).rolling(n, min_periods=n).std(ddof=0)


def rolling_percentile(x: pd.Series, n: int) -> pd.Series:
    """Rank of the current value among the last n values, in [0, 1]. Causal."""
    return x.rolling(n, min_periods=max(20, n // 10)).rank(pct=True)


def slope(x: pd.Series, n: int) -> pd.Series:
    """Least-squares slope over the last n points, normalised by the current level."""
    idx = np.arange(n, dtype=float)
    idx -= idx.mean()
    denom = (idx ** 2).sum()

    def f(w):
        return float((idx * (w - w.mean())).sum() / denom)
    return x.rolling(n, min_periods=n).apply(f, raw=True) / x


def rolling_vwap(close: pd.Series, volume: pd.Series, quote_volume: pd.Series, n: int) -> pd.Series:
    v = volume.rolling(n, min_periods=n).sum()
    return quote_volume.rolling(n, min_periods=n).sum() / v.replace(0, np.nan)


def zscore(x: pd.Series, n: int) -> pd.Series:
    m = x.rolling(n, min_periods=n).mean()
    s = x.rolling(n, min_periods=n).std(ddof=0)
    return (x - m) / s.replace(0, np.nan)
