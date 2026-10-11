"""Data-quality checks for candle tables. Problems are reported, never repaired by invention."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.timeframes import tf_ms


def find_gaps(open_time: pd.Series, timeframe: str) -> pd.DataFrame:
    """Runs of missing bars: (gap_start_ms, gap_end_ms, missing_bars)."""
    step = tf_ms(timeframe)
    t = open_time.to_numpy(dtype="int64")
    if len(t) < 2:
        return pd.DataFrame(columns=["gap_start_ms", "gap_end_ms", "missing_bars"])
    d = np.diff(t)
    idx = np.nonzero(d > step)[0]
    return pd.DataFrame({"gap_start_ms": t[idx] + step, "gap_end_ms": t[idx + 1] - step,
                         "missing_bars": (d[idx] // step - 1).astype("int64")})


def misaligned(open_time: pd.Series, timeframe: str) -> int:
    return int((open_time.to_numpy(dtype="int64") % tf_ms(timeframe) != 0).sum())


def ohlc_violations(df: pd.DataFrame) -> pd.Series:
    bad = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["volume"] < 0)
        | df[["open", "high", "low", "close"]].isna().any(axis=1)
    )
    return bad


def return_outliers(close: pd.Series, z: float = 12.0) -> pd.Series:
    """Bars whose log return is more than `z` robust (MAD) deviations from the median."""
    r = np.log(close).diff()
    med = r.median()
    mad = (r - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=close.index)
    return ((r - med).abs() / mad > z).fillna(False)


def kline_quality(df: pd.DataFrame, timeframe: str, symbol: str = "") -> dict:
    n = len(df)
    if n == 0:
        return {"symbol": symbol, "timeframe": timeframe, "rows": 0, "quality_score": 0.0}
    gaps = find_gaps(df["open_time"], timeframe)
    expected = int((df["open_time"].iloc[-1] - df["open_time"].iloc[0]) // tf_ms(timeframe)) + 1
    missing = int(gaps["missing_bars"].sum()) if len(gaps) else 0
    viol = int(ohlc_violations(df).sum())
    outl = int(return_outliers(df["close"]).sum())
    zero_vol = int((df["volume"] == 0).sum())
    dup = int(df["open_time"].duplicated().sum())
    mis = misaligned(df["open_time"], timeframe)
    missing_frac = missing / max(expected, 1)
    score = 1.0 - min(1.0, 2.0 * missing_frac + 5.0 * viol / n + 2.0 * outl / n + 0.5 * zero_vol / n
                      + 5.0 * (dup + mis) / n)
    return {"symbol": symbol, "timeframe": timeframe, "rows": n, "expected_rows": expected,
            "missing_bars": missing, "missing_frac": round(missing_frac, 6), "gap_count": len(gaps),
            "largest_gap_bars": int(gaps["missing_bars"].max()) if len(gaps) else 0,
            "ohlc_violations": viol, "return_outliers": outl, "zero_volume_bars": zero_vol,
            "duplicate_timestamps": dup, "misaligned_timestamps": mis,
            "first_utc": str(pd.to_datetime(df["open_time"].iloc[0], unit="ms", utc=True)),
            "last_utc": str(pd.to_datetime(df["open_time"].iloc[-1], unit="ms", utc=True)),
            "quality_score": round(max(0.0, score), 4)}
