"""Chronological splits with purging and embargo, plus purged walk-forward folds.

All symbols share the same calendar boundaries, so the test period is one untouched block of
time for the whole universe (no cross-asset leakage through simultaneous rows).

For a boundary b between an earlier split A and a later split B:
- **purge**: rows of A whose label horizon ends after b are removed (their labels peek into B);
- **embargo**: rows of B whose decision time is within `embargo_ms` after b are removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SplitConfig

SPLITS = ("train", "val", "calib", "test")


def boundaries(times: pd.Series, cfg: SplitConfig) -> dict[str, tuple[int, int]]:
    """[start, end) decision-time ranges for each split, by fraction of the time span."""
    fr = np.array([cfg.train_frac, cfg.val_frac, cfg.calib_frac, cfg.test_frac], dtype=float)
    if not np.isclose(fr.sum(), 1.0):
        raise ValueError(f"split fractions must sum to 1, got {fr.sum()}")
    t0, t1 = int(times.min()), int(times.max()) + 1
    edges = t0 + np.round(np.cumsum(np.r_[0, fr]) * (t1 - t0)).astype("int64")
    edges[-1] = t1
    return {name: (int(edges[i]), int(edges[i + 1])) for i, name in enumerate(SPLITS)}


def assign_splits(df: pd.DataFrame, cfg: SplitConfig, embargo_ms: int,
                  time_col: str = "decision_time", end_col: str = "label_end_time") -> pd.Series:
    """Split name per row, or None for rows removed by purge/embargo."""
    b = boundaries(df[time_col], cfg)
    t, end = df[time_col].to_numpy(), df[end_col].to_numpy()
    out = np.full(len(df), None, dtype=object)
    for name, (s, e) in b.items():
        in_split = (t >= s) & (t < e)
        keep = in_split.copy()
        if name != SPLITS[-1]:
            keep &= end <= e  # purge: label must resolve before the next split starts
        if name != SPLITS[0]:
            keep &= t >= s + embargo_ms  # embargo after the boundary
        out[keep] = name
    return pd.Series(out, index=df.index, name="split")


def walk_forward_folds(df: pd.DataFrame, n_folds: int, embargo_ms: int, time_col: str = "decision_time",
                       end_col: str = "label_end_time", min_train_frac: float = 0.4) -> list[dict]:
    """Expanding-window folds over the given (non-test) rows: train on [start, b_k), validate on
    [b_k, b_{k+1}), with purge and embargo at b_k."""
    t, end = df[time_col].to_numpy(), df[end_col].to_numpy()
    t0, t1 = int(t.min()), int(t.max()) + 1
    first = t0 + int((t1 - t0) * min_train_frac)
    edges = np.linspace(first, t1, n_folds + 1).astype("int64")
    folds = []
    for k in range(n_folds):
        b, e = int(edges[k]), int(edges[k + 1])
        train = (t < b) & (end <= b)
        val = (t >= b + embargo_ms) & (t < e) & (end <= e if k < n_folds - 1 else True)
        folds.append({"fold": k, "train_end": b, "val_start": b + embargo_ms, "val_end": e,
                      "train_idx": np.nonzero(train)[0], "val_idx": np.nonzero(val)[0]})
    return folds
