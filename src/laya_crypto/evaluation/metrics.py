"""Classification, calibration and signal-level trading metrics shared by every model.

All models are scored by the same functions on the same rows, so baseline-versus-LAYA tables
compare like with like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from ..labeling.triple_barrier import CLASSES


def _onehot(y: np.ndarray) -> np.ndarray:
    idx = np.array([CLASSES.index(v) for v in y])
    out = np.zeros((len(y), len(CLASSES)))
    out[np.arange(len(y)), idx] = 1
    return out


def ece(proba: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    """Top-1 expected calibration error with equal-width confidence bins, and the diagram data."""
    conf = proba.max(1)
    pred = np.array(CLASSES)[proba.argmax(1)]
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total, diagram = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            total += m.mean() * gap
            diagram.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": int(m.sum()),
                            "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return float(total), diagram


def coverage_curve(proba: np.ndarray, y: np.ndarray, thresholds) -> list[dict]:
    conf = proba.max(1)
    pred = np.array(CLASSES)[proba.argmax(1)]
    out = []
    for t in thresholds:
        m = conf >= t
        out.append({"threshold": t, "coverage": float(m.mean()),
                    "accuracy": float((pred[m] == y[m]).mean()) if m.any() else None})
    return out


def classification_report(y_true, proba: np.ndarray, bins: int = 15, thresholds=(0.0, 0.5, 0.6, 0.7)) -> dict:
    y = np.asarray(y_true, dtype=object)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-12, 1)
    proba = proba / proba.sum(1, keepdims=True)
    pred = np.array(CLASSES)[proba.argmax(1)]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(CLASSES), zero_division=0)
    e, diagram = ece(proba, y, bins)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(CLASSES), average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": list(CLASSES),
                             "matrix": confusion_matrix(y, pred, labels=list(CLASSES)).tolist()},
        "brier": float(((proba - _onehot(y)) ** 2).sum(1).mean()),
        "ece": e,
        "log_loss": float(log_loss(y, proba, labels=list(CLASSES))),
        "reliability": diagram,
        "coverage": coverage_curve(proba, y, thresholds),
        "pred_distribution": {c: float((pred == c).mean()) for c in CLASSES},
    }


def decision_returns(pred: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """Net return per decision from the labelled trade outcomes (costs and funding included); WAIT = 0."""
    pred = np.asarray(pred, dtype=object)
    return np.where(pred == "LONG", rows["long_net_ret"].to_numpy(),
                    np.where(pred == "SHORT", rows["short_net_ret"].to_numpy(), 0.0)).astype(float)


def block_bootstrap_mean(x: np.ndarray, block: int, n_samples: int = 1000, seed: int = 0) -> dict:
    """Moving-block bootstrap CI of the mean, for serially dependent series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    means = np.empty(n_samples)
    for i in range(n_samples):
        starts = rng.integers(0, n - block + 1, size=k)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[i] = x[idx].mean()
    return {"mean": float(x.mean()), "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)), "block": block, "samples": n_samples}


def signal_report(pred, rows: pd.DataFrame, block: int = 288, n_boot: int = 1000, seed: int = 0) -> dict:
    """Trade-level statistics of acting on every LONG/SHORT signal with the labelled execution model."""
    pred = np.asarray(pred, dtype=object)
    r = decision_returns(pred, rows)
    traded = pred != "WAIT"
    tr = r[traded]
    wins, losses = tr[tr > 0], tr[tr <= 0]
    order = np.argsort(rows["decision_time"].to_numpy(), kind="stable")
    return {
        "trades": int(traded.sum()), "trade_frequency": float(traded.mean()),
        "net_return_sum": float(tr.sum()), "expectancy": float(tr.mean()) if len(tr) else None,
        "win_rate": float((tr > 0).mean()) if len(tr) else None,
        "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
        "long_trades": int((pred == "LONG").sum()), "short_trades": int((pred == "SHORT").sum()),
        "per_decision_mean_ci": block_bootstrap_mean(r[order], block, n_boot, seed),
    }


def breakdown(pred, rows: pd.DataFrame, by: str) -> list[dict]:
    pred = np.asarray(pred, dtype=object)
    r = decision_returns(pred, rows)
    out = []
    for key, idx in rows.groupby(by).indices.items():
        p = pred[idx]
        t = p != "WAIT"
        out.append({by: key, "rows": int(len(idx)), "accuracy": float((p == rows["label"].to_numpy()[idx]).mean()),
                    "trades": int(t.sum()), "net_return_sum": float(r[idx][t].sum()),
                    "expectancy": float(r[idx][t].mean()) if t.any() else None})
    return sorted(out, key=lambda d: -d["rows"])
