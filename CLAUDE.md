# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this is

A local research project that fine-tunes the official LAYA decision model (`laya==0.4.2`) to answer
LONG / SHORT / WAIT on Binance USD-M perpetual futures market states, compares it against simple
baselines, and backtests everything under identical cost assumptions. **No live trading, no order
endpoints, no exchange keys.**

## Commands

```bash
uv sync --extra dev                       # environment (cu128 torch on Windows, CPU torch elsewhere)
uv run pytest -q                          # unit tests (synthetic data + a tiny real LAYA model)
uv run pytest -q -m network               # optional: tests that hit the public archive
uv run ruff check src tests               # lint (must pass)
uv run python -m laya_crypto.cli --help   # all pipeline commands
```

The tiny-model tests need the base tokenizer in `models/base/english/` (`laya-demo` downloads it).

## Rules that must not be broken

1. Features at row t use only data available at the close of bar t. Any new feature must keep
   `tests/test_features.py::test_full_pipeline_is_causal` passing; external series join via
   `features.pipeline.asof` with their availability time.
2. Never fill missing market data with invented values; use NaN plus `has_*` flags.
3. Splits are chronological with purge + embargo (`datasets/splits.py`). Never random-split rows.
4. The `test` split is never used for tuning, early stopping, calibration, thresholds or feature
   selection. Calibration uses `calib`; selection uses `val`.
5. Same-candle TP/SL ambiguity is resolved conservatively (stop first) in both labels and backtest.
6. Do not replace LAYA fine-tuning with heuristics, and never hard-code predictions in inference.
7. Do not commit data, checkpoints (`*.safetensors`, `*.pt`), reports or secrets (`.gitignore`).
8. Report results honestly: a baseline beating LAYA is a valid result.

## Layout

`src/laya_crypto/` packages: `ingestion`, `validation`, `features`, `labeling`, `datasets`,
`training` (incl. `laya_trainer.py`, the resumable LAYA loop built from `laya.train` parts),
`evaluation`, `backtesting`, `inference`, `optimization`, `dashboard`, `cli`. Configs in `configs/`
are validated by Pydantic models in `config.py` (unknown keys are errors). Docs in `docs/`.

## Conventions

- Timestamps: int64 epoch milliseconds, UTC, everywhere.
- Writes go through `utils/io.py` atomic helpers.
- Upstream LAYA internals used by the trainer are listed in `docs/ARCHITECTURE.md`; re-check them
  when bumping the `laya` version.
