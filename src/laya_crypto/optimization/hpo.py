"""Bounded hyper-parameter search on the VALIDATION split only (Optuna, TPE sampler).

The test split is never read here. Each LAYA trial is a full, resumable run with its own run
directory, so an interrupted search keeps the finished trials.
"""
from __future__ import annotations

import optuna

from ..config import project_path
from ..datasets.builder import load_dataset
from ..evaluation import metrics
from ..training.baselines import LightGBMBaseline, TemperatureCalibrator, y_index
from ..utils.io import write_json


def hpo_lightgbm(dataset: str, trials: int = 30, seed: int = 0) -> dict:
    df, manifest = load_dataset(dataset)
    feats = manifest["feature_columns"]
    train, val, calib = (df[df["split"] == s] for s in ("train", "val", "calib"))

    def objective(trial: optuna.Trial) -> float:
        params = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                  "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
                  "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000, log=True),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                  "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True)}
        m = LightGBMBaseline(params, seed).fit(train, val, feats)
        cal = TemperatureCalibrator().fit(m.predict_proba(calib), y_index(calib["label"]))
        return metrics.classification_report(val["label"].to_numpy(), cal.transform(m.predict_proba(val)))["log_loss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    out = {"target": "lightgbm", "metric": "val log loss", "best_value": study.best_value,
           "best_params": study.best_params, "trials": [{"params": t.params, "value": t.value} for t in study.trials]}
    write_json(project_path("reports", dataset, "hpo_lightgbm.json"), out)
    return out


def hpo_laya(dataset: str, profile: str = "deep", trials: int = 6, device: str = "auto",
             allow_cpu: bool = False, seed: int = 0) -> dict:
    from ..training.run import train_profile

    storage = f"sqlite:///{project_path('reports', dataset, 'hpo_laya.db').as_posix()}"
    project_path("reports", dataset).mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="minimize", study_name=f"laya-{profile}", storage=storage,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial: optuna.Trial) -> float:
        ov = {"encoder_lr": trial.suggest_float("encoder_lr", 5e-6, 8e-5, log=True),
              "head_lr": trial.suggest_float("head_lr", 3e-5, 5e-4, log=True),
              "loss": trial.suggest_categorical("loss", ["rlcd", "soft-ce"]),
              "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2)}
        s = train_profile(profile, dataset, resume=False, device=device, allow_cpu=allow_cpu, seeds=[seed],
                          overrides=ov, evaluate=False, tag=f"hpo{trial.number}", overwrite=True)
        from ..utils.io import read_json

        rep = read_json(project_path("models", "checkpoints", s["runs"][0], "train_report.json"))
        return float(rep["val_calibrated"]["loss"] if rep.get("val_calibrated") else rep["best_val_loss_uncalibrated"])

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if trials - done > 0:
        study.optimize(objective, n_trials=trials - done)
    out = {"target": "laya", "profile": profile, "metric": "val loss (calibrated)", "best_value": study.best_value,
           "best_params": study.best_params}
    write_json(project_path("reports", dataset, "hpo_laya.json"), out)
    return out
