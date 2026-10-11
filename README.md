# laya-crypto-trainer

Local research system that fine-tunes the official open-source **LAYA** decision model
([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), `laya==0.4.2`) to make
**LONG / SHORT / WAIT** decisions on Binance USD-M perpetual futures, and tests it against
cheaper baselines with leakage-safe splits, realistic costs and an event-driven backtest.
Everything runs on your own Windows PC. **No live trading, no orders, no exchange keys, no paid
APIs, no cloud GPUs.**

> **Azərbaycanca qısa:** Bu layihə rəsmi LAYA modelini Binance futures-un tarixi datası üzərində
> lokal olaraq fine-tune edir, onu sadə modellərlə (LightGBM, XGBoost və s.) eyni şərtlərlə
> müqayisə edir və komissiya, slippage və funding daxil olmaqla backtest edir. Quraşdırma:
> `scripts\setup.ps1`, sonra `scripts\smoke_test.ps1`. Real ticarət yoxdur. Model qiyməti
> dəqiq proqnoz etmir. Sadə modelin LAYA-dan yaxşı çıxması da etibarlı nəticədir.

## Quick start (PowerShell)

```powershell
git clone https://github.com/fuadqemberov/crypto-laya.git D:\Workspace\laya-crypto-trainer
cd D:\Workspace\laya-crypto-trainer
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # uv, Python 3.12, CUDA torch, hardware report, base LAYA
powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1     # tests + small real end-to-end run
powershell -ExecutionPolicy Bypass -File scripts\launch_dashboard.ps1   # http://127.0.0.1:8765
powershell -ExecutionPolicy Bypass -File scripts\start_bot.ps1          # data -> models -> best model -> paper bot
```

Full details: [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md).

**Google Colab (GPU):** open [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) and run all
cells. Data, checkpoints, resume states and reports are kept on Google Drive, and every cell can be
re-run after a disconnect without repeating finished work. See [docs/COLAB.md](docs/COLAB.md).

## The full workflow

```powershell
# 1. hardware + base model
uv run python -m laya_crypto.cli hardware
uv run python -m laya_crypto.cli laya-demo

# 2. data (point-in-time top-100 universe incl. delisted contracts, then all its data)
uv run python -m laya_crypto.cli universe --top-n 100
uv run python -m laya_crypto.cli download
uv run python -m laya_crypto.cli exchange-info          # current contract filters (snapshot)
uv run python -m laya_crypto.cli validate --timeframe 5m

# 3. dataset: causal features, cost-aware triple-barrier labels, purged chronological splits
uv run python -m laya_crypto.cli dataset --name main

# 4. baselines first
uv run python -m laya_crypto.cli baselines --dataset main
uv run python -m laya_crypto.cli walk-forward --dataset main --model lightgbm
uv run python -m laya_crypto.cli ablation --dataset main

# 5. LAYA
uv run python -m laya_crypto.cli train --profile smoke --dataset main
uv run python -m laya_crypto.cli train --profile research --dataset main
uv run python -m laya_crypto.cli train --profile deep --dataset main --resume      # after an interruption
uv run python -m laya_crypto.cli stop                                              # graceful stop
uv run python -m laya_crypto.cli select-best --dataset main --profile research

# 6. evaluation and backtests (identical rules for every model)
uv run python -m laya_crypto.cli evaluate --dataset main                           # reports\main\comparison.md
uv run python -m laya_crypto.cli backtest --dataset main --model lightgbm --split test
uv run python -m laya_crypto.cli backtest --dataset main --model laya-<run_id> --split test

# 7. paper trading: the model trades a simulated account on live data (no orders, no keys)
uv run python -m laya_crypto.cli paper run
uv run python -m laya_crypto.cli paper report

# 8. inference
uv run python -m laya_crypto.cli predict --symbol BTCUSDT ETHUSDT
uv run python -m laya_crypto.cli benchmark --symbol BTCUSDT
uv run python -m laya_crypto.cli dashboard
```

## What is in the box

| area | implementation | docs |
|---|---|---|
| hardware inspection and tier selection | `hardware/detect.py` | [HARDWARE_REQUIREMENTS](docs/HARDWARE_REQUIREMENTS.md) |
| Binance archive: listing-driven, checksum-verified, resumable, incremental | `ingestion/` | [DATA_PIPELINE](docs/DATA_PIPELINE.md) |
| survivorship-free monthly universe | `ingestion/universe.py` | [DATA_PIPELINE](docs/DATA_PIPELINE.md) |
| data quality (gaps, OHLC, outliers, score) | `validation/quality.py` | [DATA_PIPELINE](docs/DATA_PIPELINE.md) |
| ~98 causal features in 8 groups incl. SMC, derivatives, 1h/4h, regimes, BTC/ETH context | `features/` | [FEATURE_ENGINEERING](docs/FEATURE_ENGINEERING.md) |
| triple-barrier labels with costs, funding, conservative same-candle handling; return/TP/SL/MFE/MAE targets | `labeling/` | [LABELING](docs/LABELING.md) |
| purged + embargoed chronological splits, walk-forward folds, leakage checks, immutable manifests | `datasets/` | [MODEL_EVALUATION](docs/MODEL_EVALUATION.md) |
| baselines: always-WAIT, trend following, logistic regression, LightGBM, XGBoost | `training/baselines.py` | [MODEL_EVALUATION](docs/MODEL_EVALUATION.md) |
| LAYA fine-tuning (RLCD / soft-CE, full or frozen encoder, AMP, checkpointing), resumable at update level, chronological calibration | `training/laya_trainer.py` | [TRAINING_GUIDE](docs/TRAINING_GUIDE.md), [CHECKPOINT_RESUME](docs/CHECKPOINT_RESUME.md) |
| metrics, calibration, block bootstrap, comparison report | `evaluation/` | [MODEL_EVALUATION](docs/MODEL_EVALUATION.md) |
| event-driven portfolio backtest with risk veto | `backtesting/` | [BACKTESTING](docs/BACKTESTING.md) |
| inference (cached + live public data, abstention, staleness) | `inference/` | [INFERENCE](docs/INFERENCE.md) |
| Optuna search on validation only | `optimization/hpo.py` | [TRAINING_GUIDE](docs/TRAINING_GUIDE.md) |
| local dashboard (FastAPI + static UI, real job control) | `dashboard/` | below |
| how it fits together, LAYA APIs used, what was extended and why | — | [ARCHITECTURE](docs/ARCHITECTURE.md) |
| problems and fixes | — | [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) |
| paper-trading bot (live data, simulated account, risk manager) | `trading/paper.py` | [PAPER_TRADING](docs/PAPER_TRADING.md) |
| Colab with results on Google Drive | `notebooks/colab_train.ipynb` | [COLAB](docs/COLAB.md) |

Dashboard tabs: Hardware (live CPU/RAM/GPU), Data (coverage, quality, datasets), Training (start /
resume / graceful stop of a real subprocess, one job at a time; loss, LR, throughput, ETA, VRAM per
run; select best), Experiments (val and test tables, reliability, seeds, walk-forward, ablation),
Backtest (equity, drawdown, trades, by symbol/regime), Inference, Logs. Bound to 127.0.0.1.

## Verified in the development environment

These were executed while building the project. The development container was **not** your PC:
Linux, 4-vCPU Xeon @ 2.8 GHz, 15.7 GB RAM, **no GPU** (CPU-only PyTorch 2.14.1). Run
`hardware` on your machine for your own report.

See [docs/VERIFICATION.md](docs/VERIFICATION.md) for the exact commands and numbers.

## Research integrity

No future information in inputs (tested), no invented data, no random splits of overlapping rows,
the test split is never used for any choice, costs are always paid, same-candle TP/SL counts as the
stop, LAYA fine-tuning is never replaced by a heuristic, and inference never returns a hard-coded
prediction. A classifier probability is not a win probability. If LAYA does not beat LightGBM on
your data, that is the result.

## License and attribution

LAYA is Apache-2.0 (Convai Innovations); see [NOTICE](NOTICE). Market data © Binance, public archive.
