"""Scoring of one model's predictions on validation and (once) on the untouched test split.

Predictions are stored per model under artifacts/predictions/<dataset>/<model>.parquet with the
calibrated class probabilities, so backtests and comparison tables all read the same numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EvaluationConfig, load_evaluation_config, project_path
from ..labeling.triple_barrier import CLASSES
from ..utils.io import read_json, write_json, write_parquet
from . import metrics

PROB_COLS = [f"p_{c}" for c in CLASSES]


def predictions_dir(dataset: str) -> Path:
    return project_path("artifacts", "predictions", dataset)


def apply_abstention(proba: np.ndarray, min_conf: float) -> np.ndarray:
    pred = np.array(CLASSES, dtype=object)[proba.argmax(1)]
    pred[proba.max(1) < min_conf] = "WAIT"
    return pred


def save_predictions(dataset: str, model: str, rows: pd.DataFrame, proba: np.ndarray, extra: dict | None = None) -> Path:
    out = rows[["symbol", "open_time", "decision_time", "split", "label"]].copy()
    for i, c in enumerate(PROB_COLS):
        out[c] = proba[:, i]
    out["pred"] = np.array(CLASSES, dtype=object)[proba.argmax(1)]
    for k, v in (extra or {}).items():
        out[k] = v
    path = predictions_dir(dataset) / f"{model}.parquet"
    write_parquet(path, out)
    return path


def score_split(proba: np.ndarray, rows: pd.DataFrame, ecfg: EvaluationConfig | None = None) -> dict:
    ecfg = ecfg or load_evaluation_config()
    pred = apply_abstention(proba, ecfg.min_confidence_trade)
    return {
        "classification": metrics.classification_report(rows["label"].to_numpy(), proba, ecfg.ece_bins,
                                                        ecfg.abstain_thresholds),
        "signals": metrics.signal_report(pred, rows, ecfg.bootstrap_block, ecfg.bootstrap_samples, ecfg.seed),
        "by_symbol": metrics.breakdown(pred, rows, "symbol"),
        "by_regime": metrics.breakdown(pred, rows, "regime_trend") if "regime_trend" in rows else [],
    }


def score_model(dataset: str, model: str, rows: pd.DataFrame, proba: np.ndarray, info: dict | None = None,
                include_test: bool = True) -> dict:
    """Score val/calib (model selection) and test (final report). Writes reports/<dataset>/<model>.json."""
    ecfg = load_evaluation_config()
    rep = {"model": model, "dataset": dataset, "info": info or {}, "splits": {}}
    for split in ("train", "val", "test") if include_test else ("train", "val"):
        m = (rows["split"] == split).to_numpy()
        if m.any():
            rep["splits"][split] = score_split(proba[m], rows[m].reset_index(drop=True), ecfg)
    rep["note"] = ("train = in-sample; val = model selection; test = untouched final period, reported once "
                   "and never used for tuning, calibration or early stopping")
    write_json(project_path("reports", dataset, f"{model}.json"), rep)
    return rep


def comparison_table(dataset: str) -> pd.DataFrame:
    rows = []
    for p in sorted(project_path("reports", dataset).glob("*.json")):
        rep = read_json(p)
        if not isinstance(rep, dict) or "splits" not in rep:
            continue
        for split in ("val", "test"):
            s = rep["splits"].get(split)
            if not s:
                continue
            c, g = s["classification"], s["signals"]
            ci = g["per_decision_mean_ci"]
            rows.append({"model": rep["model"], "split": split, "n": c["n"], "accuracy": c["accuracy"],
                         "balanced_accuracy": c["balanced_accuracy"], "macro_f1": c["macro_f1"],
                         "log_loss": c["log_loss"], "brier": c["brier"], "ece": c["ece"], "trades": g["trades"],
                         "expectancy": g["expectancy"], "win_rate": g["win_rate"],
                         "profit_factor": g["profit_factor"], "net_return_sum": g["net_return_sum"],
                         "per_decision_ci_low": ci["ci_low"], "per_decision_ci_high": ci["ci_high"]})
    return pd.DataFrame(rows)
