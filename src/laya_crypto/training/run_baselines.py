"""Train and score every baseline on a dataset."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import project_path
from ..datasets.builder import load_dataset
from ..evaluation.scoring import save_predictions, score_model
from ..utils.logging import get_logger
from .baselines import REGISTRY, TemperatureCalibrator, y_index

log = get_logger("baselines")


def run_baselines(dataset: str, models: list[str] | None = None, feature_groups: list[str] | None = None,
                  seed: int = 0, tag: str = "") -> dict:
    df, manifest = load_dataset(dataset)
    features = manifest["feature_columns"]
    if feature_groups:  # ablation: keep only columns of the requested groups
        groups = manifest.get("feature_groups") or {}
        features = [c for g in feature_groups for c in groups.get(g, [])]
        if not features:
            raise ValueError(f"no features in groups {feature_groups}; available: {sorted(groups)}")
    train, val, calib = (df[df["split"] == s] for s in ("train", "val", "calib"))
    results = {}
    for name in models or list(REGISTRY):
        t0 = time.time()
        cls = REGISTRY[name]
        model = cls(seed=seed) if name in ("lightgbm", "xgboost") else cls()
        model.fit(train, val, features)
        proba = model.predict_proba(df)
        info = {"features": len(features), "fit_seconds": round(time.time() - t0, 1), "seed": seed}
        if model.learned and len(calib):
            cal = TemperatureCalibrator().fit(model.predict_proba(calib), y_index(calib["label"]))
            proba = cal.transform(proba)
            info["temperature"] = cal.T
            model.calibrator = cal
        mname = name + tag
        model.save(project_path("artifacts", "baselines", dataset, f"{mname}.pkl"))
        save_predictions(dataset, mname, df, proba)
        rep = score_model(dataset, mname, df, proba, info)
        v = rep["splits"]["val"]
        log.info("%-20s val acc %.4f  logloss %.4f  expectancy %s  (%.1fs)", mname, v["classification"]["accuracy"],
                 v["classification"]["log_loss"], v["signals"]["expectancy"], info["fit_seconds"])
        results[mname] = rep
    return results


def label_prior(df: pd.DataFrame) -> np.ndarray:
    return df["label"].value_counts(normalize=True).reindex(["LONG", "SHORT", "WAIT"]).fillna(0).to_numpy()
