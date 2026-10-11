"""Chronological splits, purge/embargo, walk-forward folds, leakage checks, serialization, LAYA schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import tokenizer_available
from laya_crypto.config import SplitConfig
from laya_crypto.datasets import leakage
from laya_crypto.datasets.serialize import QUESTION, QUESTION_ID, laya_row, serialize_row
from laya_crypto.datasets.splits import assign_splits, boundaries, walk_forward_folds

STEP = 300_000
H = 24


def rows(n=4000, symbols=("A", "B")):
    out = []
    for s in symbols:
        t = np.arange(n, dtype="int64") * STEP
        out.append(pd.DataFrame({"symbol": s, "open_time": t, "decision_time": t + STEP,
                                 "label_end_time": t + (H + 1) * STEP, "bar_ms": STEP,
                                 "label": np.random.default_rng(0).choice(["LONG", "SHORT", "WAIT"], n),
                                 "f1": np.random.default_rng(1).normal(size=n)}))
    return pd.concat(out, ignore_index=True)


def test_splits_are_chronological_purged_and_embargoed():
    df = rows()
    cfg = SplitConfig()
    df["split"] = assign_splits(df, cfg, embargo_ms=H * STEP)
    b = boundaries(df["decision_time"], cfg)
    kept = df[df["split"].notna()]
    order = ["train", "val", "calib", "test"]
    for a, c in zip(order, order[1:], strict=False):
        assert kept.loc[kept.split == a, "label_end_time"].max() <= kept.loc[kept.split == c, "decision_time"].min()
        assert kept.loc[kept.split == c, "decision_time"].min() >= b[c][0] + H * STEP
    # same calendar boundary for every symbol (no cross-asset leakage)
    for s in order:
        g = kept[kept.split == s].groupby("symbol")["decision_time"]
        assert g.min().nunique() == 1
    assert leakage.check(kept, ["f1"])["ok"]


def test_leakage_check_catches_overlap():
    df = rows()
    df["split"] = assign_splits(df, SplitConfig(), embargo_ms=0)
    df = df[df["split"].notna()].copy()
    # move one val row's horizon end beyond the next split: simulated leak
    i = df.index[df["split"] == "train"][-1]
    df.loc[i, "label_end_time"] = df.loc[df["split"] == "val", "decision_time"].min() + 10 * STEP
    assert not leakage.check(df, ["f1"])["ok"]
    # a target column passed as a feature is refused
    clean = rows()
    clean["split"] = assign_splits(clean, SplitConfig(), embargo_ms=H * STEP)
    clean = clean[clean["split"].notna()].assign(long_net_ret=0.0)
    rep = leakage.check(clean, ["f1", "long_net_ret"])
    assert not rep["ok"] and rep["checks"]["no_target_columns_in_features"]["detail"] == ["long_net_ret"]


def test_walk_forward_folds_do_not_overlap():
    df = rows(3000, ("A",))
    for f in walk_forward_folds(df, 4, H * STEP):
        tr, va = df.iloc[f["train_idx"]], df.iloc[f["val_idx"]]
        assert tr["label_end_time"].max() <= f["train_end"] <= va["decision_time"].min() - H * STEP


def test_serialization_deterministic_and_marks_missing():
    r = {"rsi": 61.234, "atr_pct": 0.0072, "htf_1h_trend": 1.0, "funding_rate": np.nan, "regime_trend": "BULL_TREND",
         "regime_vol": "HIGH_VOL"}
    s1 = serialize_row(r, "BTCUSDT", "5m")
    assert s1 == serialize_row(dict(r), "BTCUSDT", "5m")
    assert "RSI=61.2" in s1 and "ATR_PCT=0.720" in s1 and "TREND_1H=BULLISH" in s1 and "FUNDING_BP=NA" in s1
    assert s1.startswith("SYMBOL=BTCUSDT\nTIMEFRAME=5m")


@pytest.mark.skipif(not tokenizer_available(), reason="LAYA tokenizer not downloaded")
def test_rows_are_valid_laya_training_items():
    from laya.train import items_from_rows, to_internal
    from transformers import AutoTokenizer

    from tiny_laya import BASE

    tok = AutoTokenizer.from_pretrained(str(BASE / "tokenizer"))
    state = serialize_row({"rsi": 50.0, "regime_trend": "SIDEWAYS", "regime_vol": "NORMAL_VOL"}, "ETHUSDT", "5m")
    hard = [laya_row(state, lab) for lab in ("LONG", "SHORT", "WAIT")]
    soft = [laya_row(state, soft={"LONG": 0.2, "SHORT": 0.3, "WAIT": 0.5})]
    items, skipped = items_from_rows(tok, hard + soft, 512, 128)
    assert len(items) == 4 and not skipped
    assert items[0]["target"] == [1.0, 0.0, 0.0] and np.allclose(items[3]["target"], [0.2, 0.3, 0.5])
    assert list(to_internal(QUESTION_ID, QUESTION)["crit"]) == ["LONG", "SHORT", "WAIT"]
