"""Market-regime labels used as model inputs. Causal by construction.

`rule_regime` reads only indicator values at bar t. `GMMRegime` is fitted on the training period
alone and then applied row by row; the fitted model never sees validation or test rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_REGIMES = ("BULL_TREND", "BEAR_TREND", "SIDEWAYS")
VOL_REGIMES = ("HIGH_VOL", "NORMAL_VOL", "LOW_VOL")


def rule_regime(adx: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series, ema_slope: pd.Series,
                vol_pct: pd.Series, ret_z: pd.Series, adx_trend: float = 25.0, vol_high: float = 0.8,
                vol_low: float = 0.2, abnormal_z: float = 6.0) -> pd.DataFrame:
    bull = (adx >= adx_trend) & (ema_fast > ema_slow) & (ema_slope > 0)
    bear = (adx >= adx_trend) & (ema_fast < ema_slow) & (ema_slope < 0)
    trend = np.select([bull, bear], ["BULL_TREND", "BEAR_TREND"], "SIDEWAYS")
    trend = pd.Series(trend, index=adx.index).where(adx.notna() & ema_slow.notna(), "UNKNOWN")
    vol = np.select([vol_pct >= vol_high, vol_pct <= vol_low], ["HIGH_VOL", "LOW_VOL"], "NORMAL_VOL")
    vol = pd.Series(vol, index=adx.index).where(vol_pct.notna(), "UNKNOWN")
    # abnormal: an extreme return in the last 12 bars
    abnormal = ret_z.abs().rolling(12, min_periods=1).max() >= abnormal_z
    return pd.DataFrame({"regime_trend": trend, "regime_vol": vol, "regime_abnormal": abnormal.astype(np.int8)})


class GMMRegime:
    """Unsupervised regime detector: Gaussian mixture over (return, volatility, trend) features."""

    FEATURES = ("ret_12", "realized_vol_48", "ema_slope_21", "adx")

    def __init__(self, n_components: int = 4, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.model = None
        self.mean_ = None
        self.std_ = None

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return df[list(self.FEATURES)].to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame) -> GMMRegime:
        from sklearn.mixture import GaussianMixture

        x = self._x(train)
        x = x[np.isfinite(x).all(axis=1)]
        self.mean_, self.std_ = x.mean(0), x.std(0) + 1e-12
        self.model = GaussianMixture(self.n_components, covariance_type="full", random_state=self.seed)
        self.model.fit((x - self.mean_) / self.std_)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("GMMRegime.fit must be called on training data first")
        x = self._x(df)
        ok = np.isfinite(x).all(axis=1)
        out = np.full(len(df), -1)
        if ok.any():
            out[ok] = self.model.predict((x[ok] - self.mean_) / self.std_)
        return pd.Series(out, index=df.index, name="regime_gmm")
