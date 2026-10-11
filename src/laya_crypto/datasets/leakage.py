"""Leakage checks run on every built dataset. A failed check stops the build."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .splits import SPLITS


def _hashes(texts: pd.Series) -> pd.Series:
    return texts.map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def check(df: pd.DataFrame, feature_cols: list[str], state_col: str = "state") -> dict:
    """Returns a report; `report["ok"]` is False when a hard check fails."""
    rep: dict = {"checks": {}, "ok": True}

    def record(name: str, passed: bool, detail, hard: bool = True):
        rep["checks"][name] = {"passed": bool(passed), "detail": detail, "hard": hard}
        if hard and not passed:
            rep["ok"] = False

    present = [s for s in SPLITS if (df["split"] == s).any()]
    # 1. every label resolves before the next split starts, and splits are time-ordered
    for a, b in zip(present, present[1:], strict=False):
        a_end = int(df.loc[df["split"] == a, "label_end_time"].max())
        b_start = int(df.loc[df["split"] == b, "decision_time"].min())
        record(f"purge_{a}_{b}", a_end <= b_start, {"max_label_end_" + a: a_end, "min_decision_" + b: b_start})
    # 2. a (symbol, time) row is in exactly one split
    dup = df.duplicated(["symbol", "open_time"]).sum()
    record("unique_symbol_time", dup == 0, int(dup))
    # 3. features never come from the future: decision_time == open_time + timeframe for every row
    if "bar_ms" in df:
        bad = int((df["decision_time"] != df["open_time"] + df["bar_ms"]).sum())
        record("decision_time_is_bar_close", bad == 0, bad)
    # 4. identical serialized states across splits (exact duplicates)
    if state_col in df:
        h = _hashes(df[state_col])
        cross = {}
        for a in present:
            for b in present:
                if a < b:
                    ha, hb = set(h[df["split"] == a]), set(h[df["split"] == b])
                    cross[f"{a}-{b}"] = len(ha & hb)
        test_dups = sum(v for k, v in cross.items() if "test" in k)
        record("exact_state_duplicates_vs_test", test_dups == 0, cross, hard=False)
    # 5. near duplicates: rounded feature vectors shared between train and test
    if feature_cols and {"train", "test"} <= set(present):
        x = df[feature_cols].astype(float).round(3).fillna(-999.0)
        key = pd.util.hash_pandas_object(x, index=False)
        near = len(set(key[df["split"] == "train"]) & set(key[df["split"] == "test"]))
        record("near_duplicate_features_train_test", near <= max(5, int(0.001 * (df["split"] == "test").sum())),
               near, hard=False)
    # 6. no label/target column was used as a feature
    forbidden = [c for c in feature_cols if c.startswith(("label", "long_", "short_", "fwd_ret", "mfe", "mae", "p_"))]
    record("no_target_columns_in_features", not forbidden, forbidden)
    rep["rows_by_split"] = {s: int((df["split"] == s).sum()) for s in present}
    rep["label_dist_by_split"] = {s: df.loc[df["split"] == s, "label"].value_counts(normalize=True).round(4).to_dict()
                                  for s in present}
    return rep


def fit_scaler_train_only(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std from TRAIN rows only, for models that need scaled inputs."""
    x = df.loc[df["split"] == "train", cols].astype(float)
    return x.mean().to_numpy(), x.std(ddof=0).replace(0, 1.0).fillna(1.0).to_numpy()
