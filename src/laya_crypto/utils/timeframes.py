"""Timeframe arithmetic. All timestamps in this project are UTC epoch milliseconds or tz-aware UTC."""
from __future__ import annotations

import pandas as pd

TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def tf_ms(tf: str) -> int:
    try:
        return TIMEFRAME_MS[tf]
    except KeyError:
        raise ValueError(f"unsupported timeframe {tf!r}; use one of {sorted(TIMEFRAME_MS)}") from None


def tf_minutes(tf: str) -> int:
    return tf_ms(tf) // 60_000


def horizon_bars(minutes: int, tf: str) -> int:
    m = tf_minutes(tf)
    if minutes % m:
        raise ValueError(f"horizon {minutes}m is not a multiple of timeframe {tf}")
    return minutes // m


def to_utc_ms(values) -> pd.Series:
    """Normalise epoch values of unknown unit (s/ms/us/ns) to int64 milliseconds.

    Binance switched some archives to microseconds in 2025; the unit is detected by magnitude
    per value so mixed files are still handled.
    """
    s = pd.Series(values).astype("int64")
    out = s.copy()
    out[s >= 10**17] = s[s >= 10**17] // 1_000_000  # ns
    mask_us = (s >= 10**14) & (s < 10**17)
    out[mask_us] = s[mask_us] // 1_000
    mask_s = s < 10**11
    out[mask_s] = s[mask_s] * 1_000
    return out


def ms_to_ts(ms) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True)
