# Architecture

```
Binance public archive ──► ingestion ──► data/raw (verified zips) ──► data/processed (Parquet, UTC ms)
 (data.binance.vision)        │                                              │
                              └─ universe (point-in-time top-N) ◄────────────┤ 1d klines
                                                                             ▼
                     features (causal) ──► labeling (triple barrier, costs) ──► datasets
                                                                                │ purged chronological splits,
                                                                                │ leakage checks, manifest,
                                                                                │ LAYA JSONL / CSV
                       ┌────────────────────────────────────────────────────────┤
                       ▼                                                        ▼
            baselines (WAIT, trend, LR, LightGBM, XGBoost)        LAYA fine-tuning (resumable)
                       │                                                        │ calibration on `calib`
                       └──────────► artifacts/predictions/<dataset>/<model>.parquet ◄┘
                                                │
                    evaluation (metrics, bootstrap, walk-forward, ablation) · backtesting (event-driven)
                                                │
                              reports/<dataset>/… ──► dashboard (FastAPI, 127.0.0.1) · inference CLI
```

## Packages (`src/laya_crypto/`)

| package | responsibility |
|---|---|
| `config.py` | Pydantic models for every `configs/*.yaml`; unknown keys are errors |
| `hardware/` | machine inspection, resource profile selection |
| `ingestion/` | archive listing/download/verification (`archive.py`), parsers, Parquet store + manifest, point-in-time universe, public live endpoints |
| `validation/` | gaps, OHLC consistency, outliers, quality score |
| `features/` | indicators, SMC/market structure, higher-timeframe joins, regimes, the feature pipeline |
| `labeling/` | triple-barrier labels, regression targets, soft targets |
| `datasets/` | serialization for LAYA, splits, leakage checks, dataset builder |
| `training/` | base checkpoint fetch, baselines, resumable LAYA trainer, profile orchestration |
| `evaluation/` | metrics, scoring, comparison report, walk-forward, ablation |
| `backtesting/` | event-driven portfolio engine and runner |
| `inference/` | the one LAYA inference path (`laya_model.py`) and the live/cached predictor |
| `optimization/` | Optuna searches on validation only |
| `dashboard/` | FastAPI app, job manager, static single-page UI |
| `cli/` | `python -m laya_crypto.cli …` |

## LAYA integration (verified, not assumed)

- Package: `laya==0.4.2` from PyPI; upstream repository inspected at commit
  `68804629e8ccd9d616d48a40e87de9fabbeae069` (2026-10-10). License Apache-2.0, see `NOTICE`.
- Base checkpoints: `english` = `convaiinnovations/laya` at revision
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (the revision `laya.revisions.PINNED_REVISIONS`
  pins). Encoder `answerdotai/ModernBERT-large` architecture, 421,293,827 parameters, 842 MB fp16
  safetensors. `multilingual` and `typed-decisions` are also selectable with `base:`.
- APIs used: `laya.load`, `Agent.predict` / `predict_batch`; from `laya.train`: `TrainConfig`
  conventions, `load_checkpoint`, `items_from_rows`, `encode_item`, `draw_option_order`,
  `rlcd_loss`, `soft_ce_loss`, `sigma_at`, `optimizer_updates`, `calibration_records`,
  `evaluate_records`, `calibration_report`, `fit_abstention_thresholds`, `prior_collapse_message`,
  `save_checkpoint`; `laya.calibrate.fit_temperature_map`; `laya.common.collate_items`.
- Data schema: the JSONL `{state, questions, expected|gold}` rows `laya-train` reads; the
  question is one `choice` over LONG / SHORT / WAIT. The files in
  `data/datasets/<name>/laya/` can also be fed to the official CLI unchanged, e.g.
  `uv run laya-train --data data\datasets\main\laya\train.jsonl --base english --out models\official-run`.
- **Extension, and why**: upstream `finetune` saves only fp16 weights once per epoch and holds out
  a *random* calibration slice. The project trainer (`training/laya_trainer.py`) reproduces
  `train_model` step for step but adds full resume state at optimizer-update boundaries and fits
  temperatures on the chronological `calib` split. The saved checkpoints use upstream's
  `save_checkpoint` layout and load with plain `laya.load`.
- Not used because not supported/verified upstream: LoRA/QLoRA, FSDP, FlashAttention kernels,
  multi-GPU DDP (the upstream notebook's DDP script is Kaggle-specific). LAYA's optional
  TileLang fast path (`laya[fast]`) is not installed.

## Determinism and provenance

- Every dataset has an immutable `manifest.json` with SHA-256 of its files, the configs used, and
  the hashes of the universe and data manifests it was built from; `load_dataset` refuses a dataset
  whose files changed.
- Every training run has a `fingerprint.json` (profile, seed, dataset hashes, base revision, LAYA
  version); resume refuses a different fingerprint.
- Fine-tuned checkpoints carry the fingerprint, profile, seed and calibration report inside
  `rl_agent_config.json["training"]["laya_crypto"]`, and inference reports it as `model_version`.
