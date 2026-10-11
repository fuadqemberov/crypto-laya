"""Triple-barrier labels on hand-built price paths."""
from __future__ import annotations

import numpy as np
import pandas as pd

from laya_crypto.config import CostsConfig, LabelingConfig
from laya_crypto.labeling.triple_barrier import OUTCOME_SL, OUTCOME_TP, funding_between, label_frame

STEP = 300_000


def frame(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"]).astype(float)
    df["open_time"] = np.arange(len(df), dtype="int64") * STEP
    return df


def cfg(**kw):
    base = dict(method="triple_barrier_pct", tp_pct=0.02, sl_pct=0.01, max_hold_bars=3, min_barrier_pct=0.0,
                costs=CostsConfig(taker_fee=0.0, maker_fee=0.0, half_spread_bps=0, slippage_bps=0, include_funding=False))
    base.update(kw)
    return LabelingConfig(**base)


FLAT = [100, 100.2, 99.8, 100]


def test_long_take_profit_first():
    df = frame([FLAT, [100, 100.5, 99.5, 100], [100, 102.5, 99.5, 102], FLAT, FLAT, FLAT])
    lab = label_frame(df, np.full(len(df), 0.01), cfg())
    assert lab.loc[0, "label"] == "LONG" and lab.loc[0, "long_outcome"] == OUTCOME_TP
    assert lab.loc[0, "long_exit_bars"] == 2


def test_short_take_profit_first():
    df = frame([FLAT, [100, 100.5, 99.5, 100], [100, 100.5, 97.5, 98], FLAT, FLAT, FLAT])
    lab = label_frame(df, np.full(len(df), 0.01), cfg())
    assert lab.loc[0, "label"] == "SHORT"


def test_same_candle_ambiguity_is_conservative():
    df = frame([FLAT, [100, 102.5, 98.5, 100], FLAT, FLAT, FLAT, FLAT])
    lab = label_frame(df, np.full(len(df), 0.01), cfg())
    assert lab.loc[0, "long_outcome"] == OUTCOME_SL and lab.loc[0, "long_ambiguous"] == 1
    assert lab.loc[0, "label"] == "WAIT"
    dropped = label_frame(df, np.full(len(df), 0.01), cfg(ambiguity="drop"))
    assert pd.isna(dropped.loc[0, "label"])


def test_resolve_with_one_minute_data():
    df = frame([FLAT, [100, 102.5, 98.5, 100], FLAT, FLAT, FLAT, FLAT])
    t = int(df.loc[1, "open_time"])
    one = pd.DataFrame({"open_time": t + np.arange(5) * 60_000,
                        "high": [100.5, 102.5, 101, 100, 100], "low": [99.5, 100.5, 98.5, 99, 99]})
    from laya_crypto.labeling.triple_barrier import label_frame as lf

    lab = lf(df, np.full(len(df), 0.01), cfg(ambiguity="resolve_1m"), one_minute=one)
    assert lab.loc[0, "long_outcome"] == OUTCOME_TP  # minute 2 hits TP before minute 3 hits SL


def test_timeout_is_wait_and_costs_matter():
    df = frame([FLAT] * 8)
    lab = label_frame(df, np.full(len(df), 0.01), cfg())
    assert lab.loc[0, "label"] == "WAIT" and lab.loc[0, "long_outcome"] == 0
    # a TP of 0.2% is wiped out by 0.3% round-trip costs -> WAIT, not LONG
    df2 = frame([FLAT, [100, 100.3, 99.95, 100.2], FLAT, FLAT, FLAT, FLAT])
    costly = cfg(tp_pct=0.002, sl_pct=0.001, costs=CostsConfig(taker_fee=0.0015, half_spread_bps=0, slippage_bps=0,
                                                                include_funding=False))
    lab2 = label_frame(df2, np.full(len(df2), 0.01), costly)
    assert lab2.loc[0, "long_outcome"] == OUTCOME_TP and lab2.loc[0, "long_net_ret"] < 0
    assert lab2.loc[0, "label"] == "WAIT"


def test_entry_is_next_open_and_missing_bar_invalidates():
    df = frame([FLAT, [101, 101.2, 100.8, 101], FLAT, FLAT, FLAT, FLAT])
    lab = label_frame(df, np.full(len(df), 0.01), cfg())
    assert lab.loc[0, "entry_price"] == 101
    df.loc[2, ["open", "high", "low", "close"]] = np.nan
    lab = label_frame(df, np.full(len(df), 0.01), cfg())
    assert pd.isna(lab.loc[0, "label"])
    assert lab.loc[0, "label_end_time"] == (3 + 1) * STEP


def test_long_and_short_never_both_win():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 3000)))
    o = np.r_[100, c[:-1]]
    df = frame(np.c_[o, np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998, c])
    lab = label_frame(df, np.full(len(df), 0.006), LabelingConfig(method="triple_barrier_atr", tp_atr=2, sl_atr=1,
                                                                   max_hold_bars=12, min_barrier_pct=0))
    both = (lab["long_outcome"] == OUTCOME_TP) & (lab["short_outcome"] == OUTCOME_TP)
    assert not both.any()
    assert set(lab["label"].dropna()) <= {"LONG", "SHORT", "WAIT"}


def test_funding_between_is_half_open_interval():
    f = pd.DataFrame({"funding_time": [100, 200, 300], "funding_rate": [0.1, 0.2, 0.3]})
    got = funding_between(f, np.array([100, 0, 250]), np.array([300, 99, 400]))
    assert np.allclose(got, [0.5, 0.0, 0.3])


def test_labels_reproducible():
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 500)))
    o = np.r_[100, c[:-1]]
    df = frame(np.c_[o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c])
    a = label_frame(df, np.full(len(df), 0.004), LabelingConfig())
    b = label_frame(df, np.full(len(df), 0.004), LabelingConfig())
    pd.testing.assert_frame_equal(a, b)
