# Training guide

## What is being trained

LAYA is a pretrained, non-autoregressive **decision model** (ModernBERT-large encoder plus a decision
head that scores the options of a typed question). Fine-tuning teaches it one question: given this
serialized market state, which of LONG / SHORT / WAIT does the triple-barrier label favour? This is
not training a general-purpose LLM, and nothing here generates text.

## Order of work

```powershell
uv run python -m laya_crypto.cli universe
uv run python -m laya_crypto.cli download
uv run python -m laya_crypto.cli validate
uv run python -m laya_crypto.cli dataset --name main
uv run python -m laya_crypto.cli baselines --dataset main          # always before LAYA
uv run python -m laya_crypto.cli walk-forward --dataset main --model lightgbm
uv run python -m laya_crypto.cli train --profile smoke --dataset main
uv run python -m laya_crypto.cli train --profile research --dataset main
uv run python -m laya_crypto.cli select-best --dataset main --profile research
uv run python -m laya_crypto.cli evaluate --dataset main
```

## Profiles (`configs/training.yaml`)

| | smoke | research | deep |
|---|---|---|---|
| purpose | engineering check, **no predictive claim** | meaningful local experiment | full-scale study |
| train rows | 64 (evenly spaced in time) | 60,000 | all |
| eval rows per split | 48 | 15,000 | all |
| epochs | 2 | 6 (early stopping, patience 2) | 12 (patience 3) |
| micro batch × accum | 2 × 2 | 8 × 8 (adapted to the GPU) | 8 × 8 (adapted) |
| seeds | 0 | 0, 1, 2 | 0, 1, 2 |
| checkpoint every | 4 updates | 200 updates | 500 updates |
| CPU allowed | yes | only with `--allow-cpu` | only with `--allow-cpu` |

Every knob in `TrainProfile` can be changed in the YAML: `base` (`english`, `multilingual`,
`typed-decisions` or a local checkpoint directory), `loss` (`rlcd` = LAYA's default objective, or
`soft-ce`), `freeze_encoder`, `encoder_lr`, `head_lr`, `weight_decay`, `max_len`, `head_max_len`,
`amp`, `gradient_checkpointing`, `label_smoothing`, `shuffle_options` (re-orders LONG/SHORT/WAIT per
epoch to remove position bias, as upstream recommends), `seeds`, `early_stopping_patience`.
`amp` / `gradient_checkpointing` left `null` follow the hardware report.

The number of optimizer updates is what matters, not the epoch count
(`updates = epochs × ceil(ceil(items / micro) / accum)`); every run prints and stores it. LAYA's
own fine-tuning notes report that small datasets can stall at the class prior (`ln 3` loss for
three options) and that escaping it is seed-dependent: the trainer emits upstream's collapse warning
when predictions are near-constant, and the research/deep profiles use three seeds for that reason.

## Commands

```powershell
uv run python -m laya_crypto.cli train --profile research --dataset main
uv run python -m laya_crypto.cli train --profile research --dataset main --seeds 3 4
uv run python -m laya_crypto.cli train --profile deep --dataset main --resume
uv run python -m laya_crypto.cli train --profile research --dataset main --overwrite   # discard and restart
uv run python -m laya_crypto.cli stop                    # graceful stop of running runs
uv run python -m laya_crypto.cli status
uv run python -m laya_crypto.cli hpo --dataset main --target laya --profile deep --trials 6
```

The equivalent PowerShell wrapper: `scripts\train.ps1 -Profile research -Dataset main [-Resume]`.

## Device policy

`--device auto` uses CUDA when PyTorch can. If a `research`/`deep` run finds no usable GPU it stops
with the reason and the fix; it never falls back to CPU silently. `--device cuda` without CUDA is
always an error. `--device cpu --allow-cpu` is the explicit opt-in.

## What a run writes (`models/checkpoints/<run_id>/`)

| file | content |
|---|---|
| `fingerprint.json` | profile, seed, dataset hashes, base revision, LAYA version |
| `status.json` | state (running / stopped / completed / failed), progress, pid |
| `metrics.jsonl` | per log step: epoch, optimizer update, loss, epoch mean loss, encoder/head LR, items/s, measured ETA, remaining updates, GPU memory allocated/reserved/peak, GPU utilisation; per epoch: validation loss, accuracy, ECE, Brier |
| `state.pt`, `state.json` (+ `.prev`) | resume state, see [CHECKPOINT_RESUME.md](CHECKPOINT_RESUME.md) |
| `checkpoint_latest/` | LAYA-format weights after the last epoch |
| `best_uncalibrated/` | weights of the best validation-loss epoch |
| `final/` | best epoch + temperatures fitted on `calib`, `questions.json`; loads with `laya.load` |
| `train_report.json` | losses, validation history, calibration report, collapse warning |

After training, each seed's `final/` is scored through `laya.load(...).predict_batch(...)` on the
validation and test splits (same rows, labels and metrics as the baselines) and the seed summary is
written to `reports/<dataset>/laya_<profile>_summary.json`. `select-best` copies the seed with the
lowest **validation** log loss to `models/best/`.

## Using the official CLI instead

The dataset's LAYA files are valid `laya-train` input:

```powershell
uv run laya-train --data data\datasets\main\laya\train.jsonl --eval data\datasets\main\laya\val.jsonl --base english --out models\official-main
```

That path has no mid-epoch resume and calibrates on a random slice of the training file.
