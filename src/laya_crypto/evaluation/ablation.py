"""Leave-one-group-out feature ablation on the VALIDATION split (the test split is not touched)."""
from __future__ import annotations

import numpy as np

from ..config import project_path
from ..datasets.builder import load_dataset
from ..training.baselines import REGISTRY, TemperatureCalibrator, y_index
from ..utils.io import write_json
from . import metrics


def run_ablation(dataset: str, model: str = "lightgbm") -> dict:
    df, manifest = load_dataset(dataset)
    groups = manifest["feature_groups"]
    train, val, calib = (df[df["split"] == s] for s in ("train", "val", "calib"))

    def fit_score(features):
        m = REGISTRY[model]()
        m.fit(train, val, features)
        cal = TemperatureCalibrator().fit(m.predict_proba(calib), y_index(calib["label"]))
        p = cal.transform(m.predict_proba(val))
        rep = metrics.classification_report(val["label"].to_numpy(), p)
        pred = np.array(["LONG", "SHORT", "WAIT"], dtype=object)[p.argmax(1)]
        sig = metrics.signal_report(pred, val.reset_index(drop=True), n_boot=200)
        return {"features": len(features), "log_loss": rep["log_loss"], "accuracy": rep["accuracy"],
                "macro_f1": rep["macro_f1"], "expectancy": sig["expectancy"]}

    all_feats = manifest["feature_columns"]
    out = {"model": model, "split": "val", "all": fit_score(all_feats), "without": {}, "only": {}}
    for g, cols in groups.items():
        rest = [c for c in all_feats if c not in set(cols)]
        if rest:
            out["without"][g] = fit_score(rest)
        out["only"][g] = fit_score(cols)
    base_ll = out["all"]["log_loss"]
    out["delta_log_loss_when_removed"] = {g: round(v["log_loss"] - base_ll, 5) for g, v in out["without"].items()}
    out["note"] = ("positive delta = removing the group hurts (the group helps). Small deltas are within noise; "
                   "repeat with other seeds / walk-forward folds before dropping features.")
    write_json(project_path("reports", dataset, f"ablation_{model}.json"), out)
    return out
