"""Baseline models trained on exactly the same rows, splits and costs as LAYA.

Each baseline returns class probabilities in `CLASSES` order. Learned models get the same
post-hoc calibration LAYA gets (one temperature fitted on the calibration split), so the
comparison is not tilted by calibration either.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..labeling.triple_barrier import CLASSES

Y_MAP = {c: i for i, c in enumerate(CLASSES)}


def y_index(labels) -> np.ndarray:
    return np.array([Y_MAP[v] for v in labels])


class TemperatureCalibrator:
    """p' = softmax(log p / T), T fitted by NLL on calibration rows only."""

    def __init__(self):
        self.T = 1.0

    def fit(self, proba: np.ndarray, y: np.ndarray) -> TemperatureCalibrator:
        logp = np.log(np.clip(proba, 1e-12, 1))

        def nll(log_t):
            z = logp / np.exp(log_t)
            z -= z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(len(y)), y].mean()
        res = minimize_scalar(nll, bounds=(np.log(0.1), np.log(10)), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(proba, 1e-12, 1)) / self.T
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


class Baseline:
    name = "baseline"
    learned = False

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> Baseline:
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class AlwaysWait(Baseline):
    name = "always_wait"

    def predict_proba(self, df):
        p = np.zeros((len(df), 3))
        p[:, Y_MAP["WAIT"]] = 1.0
        return p


class TrendFollowing(Baseline):
    """LONG when the 1h trend is up, price is above EMA21 and ADX >= 20; SHORT mirrored; else WAIT.
    Probabilities are the empirical label frequencies of each rule signal on TRAIN rows."""

    name = "trend_following"

    def __init__(self, adx_min: float = 20.0):
        self.adx_min = adx_min
        self.table = None

    def signal(self, df) -> np.ndarray:
        up = (df["htf_1h_trend"] == 1) & (df["ema21_dist_atr"] > 0) & (df["adx"] >= self.adx_min)
        dn = (df["htf_1h_trend"] == -1) & (df["ema21_dist_atr"] < 0) & (df["adx"] >= self.adx_min)
        return np.select([up, dn], ["LONG", "SHORT"], "WAIT")

    def fit(self, train, val, features):
        sig = self.signal(train)
        self.table = {}
        for s in CLASSES:
            lab = train["label"][sig == s]
            freq = np.array([(lab == c).mean() if len(lab) else 0.0 for c in CLASSES])
            if len(lab) == 0:
                freq = np.eye(3)[Y_MAP[s]]
            self.table[s] = freq
        return self

    def predict_proba(self, df):
        sig = self.signal(df)
        return np.stack([self.table[s] for s in sig]) if len(df) else np.zeros((0, 3))


class LogisticBaseline(Baseline):
    name = "logistic_regression"
    learned = True

    def fit(self, train, val, features):
        from sklearn.linear_model import LogisticRegression

        self.features = features
        x = train[features].astype(float)
        self.median = x.median().fillna(0.0)
        x = x.fillna(self.median)
        self.mean, self.std = x.mean(), x.std(ddof=0).replace(0, 1.0).fillna(1.0)  # TRAIN statistics only
        self.model = LogisticRegression(max_iter=2000, C=0.1)
        self.model.fit(((x - self.mean) / self.std).to_numpy(), y_index(train["label"]))
        return self

    def predict_proba(self, df):
        x = df[self.features].astype(float).fillna(self.median)
        p = self.model.predict_proba(((x - self.mean) / self.std).to_numpy())
        out = np.zeros((len(df), 3))
        out[:, self.model.classes_] = p
        return out


class LightGBMBaseline(Baseline):
    name = "lightgbm"
    learned = True

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.params = {"n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 200,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0,
                       **(params or {})}
        self.seed = seed

    def fit(self, train, val, features):
        import lightgbm as lgb

        self.features = features
        self.model = lgb.LGBMClassifier(objective="multiclass", random_state=self.seed, verbose=-1, **self.params)
        self.model.fit(train[features].astype(float), y_index(train["label"]),
                       eval_X=(val[features].astype(float),), eval_y=(y_index(val["label"]),),
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(df[self.features].astype(float))


class XGBoostBaseline(Baseline):
    name = "xgboost"
    learned = True

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.params = {"n_estimators": 2000, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 20,
                       "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.0, "tree_method": "hist",
                       **(params or {})}
        self.seed = seed

    def fit(self, train, val, features):
        import xgboost as xgb

        self.features = features
        self.model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, random_state=self.seed,
                                       early_stopping_rounds=100, eval_metric="mlogloss", **self.params)
        self.model.fit(train[features].astype(float), y_index(train["label"]),
                       eval_set=[(val[features].astype(float), y_index(val["label"]))], verbose=False)
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(df[self.features].astype(float))


REGISTRY = {c.name: c for c in (AlwaysWait, TrendFollowing, LogisticBaseline, LightGBMBaseline, XGBoostBaseline)}


def load_baseline(path: Path) -> Baseline:
    with open(path, "rb") as f:
        return pickle.load(f)
