"""Higher-timeframe context without look-ahead.

Higher-timeframe candles are built from the base candles, and only from complete groups: a 1h
candle needs all twelve 5m candles, otherwise it is missing. A higher-timeframe candle opening at T
is known at T + duration. It is joined onto a base bar only if that time is <= the base bar's close,
so a 5m bar closing at 10:55 sees the 09:00-10:00 1h candle, and the bar closing at 11:00 sees the
10:00-11:00 one. A partially formed higher-timeframe candle is never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.timeframes import tf_ms
from . import indicators as ind


def resample_complete(df: pd.DataFrame, base_tf: str, target_tf: str) -> pd.DataFrame:
    base, target = tf_ms(base_tf), tf_ms(target_tf)
    if target % base:
        raise ValueError(f"{target_tf} is not a multiple of {base_tf}")
    per = target // base
    g = df.assign(bucket=(df["open_time"] // target) * target).groupby("bucket")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(), "n": g["open"].size(),
        "first_open_time": g["open_time"].min(),
    })
    out = out[(out["n"] == per) & (out["first_open_time"] == out.index)]
    out = out.drop(columns=["n", "first_open_time"]).rename_axis("open_time").reset_index()
    out["available_time"] = out["open_time"] + target
    return out


def htf_features(htf: pd.DataFrame, prefix: str, rsi_n: int = 14, atr_n: int = 14, adx_n: int = 14) -> pd.DataFrame:
    c = htf["close"]
    e21, e50 = ind.ema(c, 21), ind.ema(c, 50)
    a = ind.atr(htf["high"], htf["low"], c, atr_n)
    adx = ind.adx(htf["high"], htf["low"], c, adx_n)["adx"]
    trend = np.select([(e21 > e50) & (c > e21), (e21 < e50) & (c < e21)], [1, -1], 0)
    out = pd.DataFrame({
        "available_time": htf["available_time"],
        f"{prefix}_trend": np.where(e50.notna(), trend, np.nan),
        f"{prefix}_rsi": ind.rsi(c, rsi_n),
        f"{prefix}_ret_1": c.pct_change(),
        f"{prefix}_ema21_dist_atr": (c - e21) / a,
        f"{prefix}_adx": adx,
        f"{prefix}_atr_pct": a / c,
    })
    return out


def join_htf(base: pd.DataFrame, htf_feats: pd.DataFrame, base_tf: str,
             max_age_ms: int | None = None) -> pd.DataFrame:
    """As-of join on availability: base close time >= higher-timeframe candle close time.

    `max_age_ms` drops a match older than that (after a data gap) instead of carrying it forward.
    """
    left = base[["open_time"]].assign(decision_time=base["open_time"] + tf_ms(base_tf))
    right = htf_feats.sort_values("available_time")
    merged = pd.merge_asof(left.sort_values("decision_time"), right, left_on="decision_time",
                           right_on="available_time", direction="backward",
                           tolerance=max_age_ms)
    merged = merged.set_index(left.sort_values("decision_time").index).reindex(base.index)
    return merged.drop(columns=["open_time", "decision_time", "available_time"])
