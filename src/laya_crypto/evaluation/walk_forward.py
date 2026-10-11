"""Purged expanding-window walk-forward evaluation over the non-test period.

The final test split is excluded entirely; folds are built from train+val+calib rows only.
"""
from __future__ import annotations

import numpy as np

from ..config import load_dataset_config, project_path
from ..datasets.builder import load_dataset
from ..datasets.splits import walk_forward_folds
from ..training.baselines import REGISTRY
from ..utils.io import write_json
from . import metrics


def walk_forward_baseline(dataset: str, model: str = "lightgbm", folds: int | None = None) -> dict:
    df, manifest = load_dataset(dataset)
    dcfg = load_dataset_config()
    pool = df[df["split"] != "test"].reset_index(drop=True)
    features = manifest["feature_columns"]
    res = []
    for f in walk_forward_folds(pool, folds or dcfg.split.walk_forward_folds, manifest["embargo_ms"]):
        tr, va = pool.iloc[f["train_idx"]], pool.iloc[f["val_idx"]]
        if len(tr) < 1000 or len(va) < 200:
            continue
        # inner early-stopping slice = last 15% of the fold's training window (time-ordered)
        cut = tr["decision_time"].quantile(0.85)
        inner_tr, inner_va = tr[tr["label_end_time"] <= cut], tr[tr["decision_time"] > cut + manifest["embargo_ms"]]
        m = REGISTRY[model]().fit(inner_tr, inner_va, features)
        p = m.predict_proba(va)
        rep = metrics.classification_report(va["label"].to_numpy(), p)
        pred = np.array(["LONG", "SHORT", "WAIT"], dtype=object)[p.argmax(1)]
        sig = metrics.signal_report(pred, va.reset_index(drop=True), n_boot=200)
        res.append({"fold": f["fold"], "train_rows": len(tr), "val_rows": len(va),
                    "val_start_ms": f["val_start"], "val_end_ms": f["val_end"], "log_loss": rep["log_loss"],
                    "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"], "expectancy": sig["expectancy"],
                    "trades": sig["trades"]})
    summary = {"model": model, "dataset": dataset, "folds": res,
               "mean_log_loss": float(np.mean([r["log_loss"] for r in res])) if res else None,
               "mean_expectancy": float(np.nanmean([r["expectancy"] for r in res if r["expectancy"] is not None]))
               if res else None, "note": "test split excluded; purge + embargo at every fold boundary"}
    write_json(project_path("reports", dataset, f"walk_forward_{model}.json"), summary)
    return summary
