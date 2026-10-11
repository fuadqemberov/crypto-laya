# Hardware requirements

Run `uv run python -m laya_crypto.cli hardware` first. It measures, rather than assumes, the OS
build, CPU, RAM, every display adapter the OS reports (any vendor), `nvidia-smi` (driver, VRAM, compute
capability, utilisation), whether PyTorch can launch a CUDA kernel, WSL status, disk free space and
sequential write speed. It writes `reports/hardware_report.json` and `reports/hardware_report.md`,
and picks a training tier:

| tier | condition (VRAM minus 1 GB reserve) | micro batch × grad accum | AMP | gradient checkpointing |
|---|---|---|---|---|
| `low_vram` | < 8 GB | 2 × 32 | if CC ≥ 7.0 | on; freeze encoder suggested below 5 GB |
| `medium_vram` | 8–16 GB | 4 × 16 | if CC ≥ 7.0 | on |
| `high_vram` | ≥ 16 GB | 8 × 8 | if CC ≥ 7.0 | on below 22 GB |
| `cpu` | no usable CUDA | 2 × 4 | off | off |

The effective batch (64) is the same in every tier, so changing GPUs does not change the
optimisation problem, only its speed. GTX 16xx cards get AMP disabled (their fp16 is slower than
fp32, as LAYA's docs note). The thresholds live in `configs/hardware.yaml`.

On a CUDA machine the trainer also probes the longest training sequences before step one and
halves the micro batch (doubling accumulation) until a forward + backward pass fits.

## What the model needs

The `english` base is ModernBERT-large sized: **421 M parameters**. A full fine-tune with AdamW
keeps fp32 weights, gradients and two optimizer moments: about 6.7 GB before activations.

- **NVIDIA ≥ 12 GB** (e.g. RTX 3060 12 GB, 4070, 4080, 4090, 5070 Ti …): full fine-tune with AMP and
  gradient checkpointing at `max_len 512`.
- **NVIDIA 4 GB (e.g. GTX 1050 / 1650)**: LAYA inference fits (fp32 weights ~1.7 GB). Training only
  with `freeze_encoder: true`, and slowly; Pascal/Turing GTX cards get no AMP speed-up. Train on
  Colab and use the laptop for inference and the paper bot.
- **NVIDIA 8 GB**: full fine-tune is tight. Use the low-VRAM tier and set `freeze_encoder: true` for
  research runs if the OOM probe reaches micro batch 1.
- **AMD / Intel GPUs**: PyTorch CUDA wheels do not use them. Training runs on the CPU (very slow,
  smoke profile only). DirectML/ROCm are not supported by this project.
- **RAM**: 16 GB minimum for the full pipeline, 32 GB recommended for 100+ symbol datasets.
- **CPU-only**: measured in the development container (4 vCPU Xeon @ 2.8 GHz, no GPU) at
  ~0.1 training items/s for the full model at `max_len 512`. The smoke profile (64 items, 2 epochs)
  works. Research and deep profiles on CPU are refused unless you pass `--allow-cpu`.

No GPU benchmark numbers are given here because none were measured on a GPU; every run records its
own measured throughput (`items_per_s`) and ETA in `models/checkpoints/<run>/metrics.jsonl`.

## Disk

| item | size |
|---|---|
| `.venv` (CUDA PyTorch) | ~5 GB on Windows |
| base checkpoint | 0.85 GB |
| 5m klines + mark/index/premium + metrics, 100 symbols, 1 year | a few GB of zips + Parquet |
| one LAYA run | ~12 GB (4.8 GB resume state + previous state + 3 checkpoint folders) |

Delete `models/checkpoints/<run>/state.pt*` after a run completes if you will not resume it.
