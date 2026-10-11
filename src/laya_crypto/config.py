"""Typed configuration loaded from configs/*.yaml.

Every pipeline stage reads one of these models, and the effective values are written into the
artifacts each stage produces, so a run can always be traced back to the configuration it used.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(os.environ.get("LAYA_CRYPTO_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = Path(os.environ.get("LAYA_CRYPTO_CONFIG_DIR", PROJECT_ROOT / "configs"))


class Strict(BaseModel):
    model_config = {"extra": "forbid"}


# ------------------------------------------------------------------------------------- data

class UniverseConfig(Strict):
    top_n: int = 100
    quote_asset: str = "USDT"
    rank_window_days: int = 30
    min_history_days: int = 60
    min_avg_daily_quote_volume: float = 10_000_000.0
    exclude_symbols: list[str] = Field(default_factory=lambda: ["USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT"])
    include_symbols: list[str] = Field(default_factory=list)  # always considered (still filtered)
    max_symbols: int | None = None  # hard cap for smoke runs


class DownloadConfig(Strict):
    concurrency: int = 8
    retries: int = 5
    backoff_base_s: float = 1.0
    timeout_s: float = 60.0
    verify_checksums: bool = True


class DatasetsConfig(Strict):
    klines: bool = True
    mark_price_klines: bool = True
    index_price_klines: bool = True
    premium_index_klines: bool = True
    funding_rate: bool = True
    metrics: bool = True  # open interest + long/short ratios (5m snapshots, daily files)
    agg_trades: bool = False  # large: enable for CVD research
    book_depth: bool = False  # daily depth-percentage snapshots, from 2023 only


class DataConfig(Strict):
    root: str = "data"
    archive_base_url: str = "https://data.binance.vision"
    listing_base_url: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    fapi_base_url: str = "https://fapi.binance.com"
    start: str = "2022-01-01"
    end: str | None = None  # None = last complete UTC day
    base_timeframe: str = "5m"
    timeframes: list[str] = Field(default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"])
    context_symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    metrics_publication_lag_ms: int = 300_000  # treat a 5m OI snapshot as known one bar later
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)

    @property
    def root_path(self) -> Path:
        p = Path(self.root)
        return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------------- features

class FeaturesConfig(Strict):
    rsi_period: int = 14
    ema_periods: list[int] = Field(default_factory=lambda: [9, 21, 50, 200])
    sma_periods: list[int] = Field(default_factory=lambda: [20, 50])
    macd: tuple[int, int, int] = (12, 26, 9)
    atr_period: int = 14
    adx_period: int = 14
    roc_periods: list[int] = Field(default_factory=lambda: [3, 12, 48])
    stoch_rsi: tuple[int, int, int, int] = (14, 14, 3, 3)
    bb_period: int = 20
    bb_std: float = 2.0
    vol_windows: list[int] = Field(default_factory=lambda: [12, 48, 288])
    vol_percentile_window: int = 2016  # 7 days of 5m bars
    rel_volume_window: int = 48
    volume_spike_z: float = 3.0
    vwap_window: int = 288
    slope_window: int = 12
    swing_lookback: int = 3  # bars on each side; a swing is confirmed `swing_lookback` bars late
    fvg_min_atr: float = 0.1
    order_block_lookback: int = 20
    sweep_lookback: int = 48
    sr_window: int = 288
    higher_timeframes: list[str] = Field(default_factory=lambda: ["1h", "4h"])
    regime_trend_adx: float = 25.0
    regime_vol_high_pct: float = 0.8
    regime_vol_low_pct: float = 0.2
    regime_abnormal_return_z: float = 6.0
    include_context: bool = True
    groups: list[str] = Field(default_factory=lambda: [
        "trend", "volatility", "volume", "structure", "derivatives", "mtf", "regime", "context"])


# --------------------------------------------------------------------------------- labeling

class CostsConfig(Strict):
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    entry_order: Literal["taker", "maker"] = "taker"
    exit_order: Literal["taker", "maker"] = "taker"
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    include_funding: bool = True

    def round_trip(self) -> float:
        fee = (self.taker_fee if self.entry_order == "taker" else self.maker_fee) + (
            self.taker_fee if self.exit_order == "taker" else self.maker_fee)
        return fee + 2 * (self.half_spread_bps + self.slippage_bps) / 10_000


class LabelingConfig(Strict):
    method: Literal["triple_barrier_atr", "triple_barrier_pct"] = "triple_barrier_atr"
    tp_atr: float = 2.0
    sl_atr: float = 1.0
    tp_pct: float = 0.01
    sl_pct: float = 0.005
    max_hold_bars: int = 24  # 2h of 5m bars
    min_barrier_pct: float = 0.002  # floor so barriers stay above round-trip costs
    ambiguity: Literal["conservative_sl", "resolve_1m", "drop"] = "conservative_sl"
    timeout_policy: Literal["wait", "sign_if_net_positive"] = "wait"
    return_horizons_min: list[int] = Field(default_factory=lambda: [15, 30, 60])
    soft_targets: bool = False
    soft_barrier_scales: list[float] = Field(default_factory=lambda: [0.75, 1.0, 1.25])
    costs: CostsConfig = Field(default_factory=CostsConfig)

    @model_validator(mode="after")
    def _tp_ge_sl(self):
        if self.method == "triple_barrier_atr" and self.tp_atr < self.sl_atr:
            raise ValueError("tp_atr < sl_atr makes LONG and SHORT labels non-exclusive")
        return self


# --------------------------------------------------------------------------------- datasets

class SplitConfig(Strict):
    train_frac: float = 0.6
    val_frac: float = 0.15
    calib_frac: float = 0.1
    test_frac: float = 0.15
    embargo_bars: int | None = None  # None = label horizon (max_hold_bars)
    walk_forward_folds: int = 4


class DatasetConfig(Strict):
    timeframe: str = "5m"
    max_rows_per_symbol: int | None = None
    sample_every: int = 1  # take every Nth bar; >1 thins overlapping labels
    wait_downsample: float = 1.0  # keep this fraction of WAIT rows in TRAIN only
    question_id: str = "decision"
    split: SplitConfig = Field(default_factory=SplitConfig)


# --------------------------------------------------------------------------------- training

class TrainProfile(Strict):
    base: str = "english"
    loss: Literal["rlcd", "soft-ce"] = "rlcd"
    freeze_encoder: bool = False
    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 8
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    max_len: int | None = 512
    head_max_len: int | None = 192
    amp: bool | None = None
    gradient_checkpointing: bool | None = None
    label_smoothing: float = 0.0
    shuffle_options: bool = True
    seeds: list[int] = Field(default_factory=lambda: [0])
    max_train_rows: int | None = None
    max_eval_rows: int | None = None
    checkpoint_every_steps: int = 200
    eval_every_epochs: int = 1
    early_stopping_patience: int | None = None
    hpo_trials: int = 0
    log_every_steps: int = 20
    oom_retry: bool = True


class TrainingConfig(Strict):
    profiles: dict[str, TrainProfile]
    output_dir: str = "models"
    allow_cpu: bool = False


# ------------------------------------------------------------------------------- evaluation

class EvaluationConfig(Strict):
    ece_bins: int = 15
    bootstrap_samples: int = 1000
    bootstrap_block: int = 288
    abstain_thresholds: list[float] = Field(default_factory=lambda: [0.0, 0.4, 0.5, 0.6, 0.7, 0.8])
    min_confidence_trade: float = 0.0
    seed: int = 7


class BacktestConfig(Strict):
    initial_equity_usdt: float = 2000.0
    risk_per_trade: float = 0.005
    max_leverage: float = 3.0
    max_positions: int = 5
    max_position_notional_frac: float = 0.5
    max_portfolio_drawdown: float = 0.25  # halt new entries beyond this drawdown
    maintenance_margin_rate: float = 0.005
    min_confidence: float = 0.0
    one_position_per_symbol: bool = True


class InferenceConfig(Strict):
    checkpoint: str = "models/best"
    timeframe: str = "5m"
    abstain_below: float = 0.45
    max_data_age_minutes: int = 15
    warmup_bars: int = 2400
    host: str = "127.0.0.1"
    port: int = 8765


class HardwareConfig(Strict):
    low_vram_gb: float = 8.0
    high_vram_gb: float = 16.0
    reserve_vram_gb: float = 1.0


def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_config() -> DataConfig:
    return DataConfig(**_load_yaml("data"))


def load_features_config() -> FeaturesConfig:
    return FeaturesConfig(**_load_yaml("features"))


def load_labeling_config() -> LabelingConfig:
    return LabelingConfig(**_load_yaml("labeling"))


def load_dataset_config() -> DatasetConfig:
    return DatasetConfig(**_load_yaml("dataset"))


def load_training_config() -> TrainingConfig:
    raw = _load_yaml("training")
    raw.setdefault("profiles", {"smoke": {}})
    return TrainingConfig(**raw)


def load_evaluation_config() -> EvaluationConfig:
    return EvaluationConfig(**_load_yaml("evaluation"))


def load_backtest_config() -> BacktestConfig:
    return BacktestConfig(**_load_yaml("backtest"))


def load_inference_config() -> InferenceConfig:
    return InferenceConfig(**_load_yaml("inference"))


def load_hardware_config() -> HardwareConfig:
    return HardwareConfig(**_load_yaml("hardware"))


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
