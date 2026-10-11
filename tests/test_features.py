"""Feature correctness and, above all, causality: no value at bar t may change when the future changes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import T0, synthetic_klines
from laya_crypto.config import FeaturesConfig
from laya_crypto.features import indicators as ind
from laya_crypto.features.mtf import htf_features, join_htf, resample_complete
from laya_crypto.features.pipeline import AuxData, asof, build_features
from laya_crypto.features.structure import confirmed_swings, structure_features


def test_rsi_extremes_and_range():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up, 14).dropna().eq(100).all()
    r = ind.rsi(synthetic_klines(500)["close"], 14).dropna()
    assert r.between(0, 100).all()


def test_ema_matches_definition():
    x = pd.Series(np.random.default_rng(0).normal(size=200)).cumsum()
    e = ind.ema(x, 10)
    a = 2 / 11
    manual = [x.iloc[0]]
    for v in x.iloc[1:]:
        manual.append(a * v + (1 - a) * manual[-1])
    assert np.allclose(e.iloc[9:], manual[9:])


def test_atr_is_mean_true_range_for_constant_range():
    df = pd.DataFrame({"high": [11.0] * 50, "low": [9.0] * 50, "close": [10.0] * 50})
    assert np.isclose(ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1], 2.0)


def test_full_pipeline_is_causal():
    """Features of row k computed from data[:k+1] equal those computed from the full history."""
    df = synthetic_klines(1500, seed=3)
    cfg = FeaturesConfig(vol_percentile_window=300, sr_window=100)
    ctx = {"BTCUSDT": synthetic_klines(1500, seed=9)[["open_time", "close"]]}
    full = build_features(df, cfg, "5m", AuxData(), ctx)
    cols = full.columns()
    for k in (700, 1001, 1499):
        part = build_features(df.iloc[: k + 1], cfg, "5m", AuxData(), {s: c.iloc[: k + 1] for s, c in ctx.items()})
        a = full.frame.loc[k, cols].astype(float)
        b = part.frame.loc[k, cols].astype(float)
        diff = [c for c in cols if not (np.isclose(a[c], b[c], equal_nan=True, rtol=1e-9, atol=1e-12))]
        assert not diff, f"look-ahead in {diff} at row {k}"


def test_gaps_become_nan_rows_not_contiguous():
    df = synthetic_klines(400).drop(index=range(200, 210))
    ff = build_features(df, FeaturesConfig(vol_percentile_window=100), "5m")
    assert len(ff.frame) == 400 and (~ff.frame["bar_present"]).sum() == 10
    # a 20-bar window ending right after the gap must not be computed from 20 contiguous-looking rows
    assert np.isnan(ff.frame.loc[212, "bb_width"])


def test_swing_confirmed_only_after_lookback():
    high = np.array([1, 2, 3, 10, 3, 2, 1, 1, 1], dtype=float)
    low = high - 0.5
    sh, _ = confirmed_swings(high, low, 3)
    assert not sh[:6].any() and sh[6]  # pivot at index 3 becomes known at index 6 (= 3 + L)


def test_bos_and_choch():
    # down-structure, then a break above the last swing high -> CHoCH up
    closes = [10, 9, 8, 9, 10, 9, 7, 6, 5, 6, 7, 6, 5, 4, 3, 4, 5, 6, 8, 11, 12]
    df = pd.DataFrame({"open": closes, "close": closes})
    df["high"] = df["close"] + 0.2
    df["low"] = df["close"] - 0.2
    st = structure_features(df, pd.Series(np.ones(len(df))), lookback=1, sweep_lookback=3, sr_window=5)
    assert st["bos_down"].sum() + st["choch_down"].sum() >= 1
    assert st["choch_up"].sum() >= 1


def test_fvg_definition():
    df = pd.DataFrame({"open": [10, 10.5, 12.0], "close": [10.4, 12, 12.5], "high": [10.5, 12.1, 13.0],
                       "low": [9.9, 10.4, 11.0]})
    st = structure_features(df, pd.Series([0.5, 0.5, 0.5]), lookback=1, sweep_lookback=1, sr_window=2)
    assert st["fvg_bull"].tolist() == [0, 0, 1]  # low[2]=11.0 > high[0]=10.5


def test_htf_alignment_uses_only_closed_candles():
    df = synthetic_klines(48)  # 4 hours of 5m bars from 00:00
    htf = resample_complete(df, "5m", "1h")
    assert len(htf) == 4 and (htf["available_time"] == htf["open_time"] + 3_600_000).all()
    feats = htf_features(htf, "h").assign(h_close=htf["close"].to_numpy())
    j = join_htf(df, feats, "5m")
    # bar 10 (00:50-00:55) closes before the first 1h candle closes -> nothing known yet
    assert np.isnan(j.loc[10, "h_close"])
    # bar 11 (00:55-01:00) closes exactly when the 00:00-01:00 candle closes -> it is known
    assert j.loc[11, "h_close"] == htf["close"].iloc[0]
    # bar 22 still sees the first hour, bar 23 the second
    assert j.loc[22, "h_close"] == htf["close"].iloc[0] and j.loc[23, "h_close"] == htf["close"].iloc[1]


def test_incomplete_htf_candle_dropped():
    df = synthetic_klines(24).drop(index=[5])
    htf = resample_complete(df, "5m", "1h")
    assert len(htf) == 1 and htf["open_time"].iloc[0] == T0 + 3_600_000  # only the complete second hour


def test_asof_respects_publication_lag():
    dec = pd.Series([1000, 2000, 3000])
    right = pd.DataFrame({"t": [900, 1900], "v": [1.0, 2.0]})
    out = asof(dec, right, "t", ["v"], lag_ms=200)
    assert np.isnan(out["v"].iloc[0]) and out["v"].iloc[1] == 1.0 and out["v"].iloc[2] == 2.0
