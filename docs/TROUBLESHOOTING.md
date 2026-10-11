# Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `CUDA was requested but PyTorch cannot use a GPU` | CPU-only torch wheel, or driver too old for cu128 | `nvidia-smi` must show CUDA ≥ 12.8; then `uv sync` (Windows pulls the cu128 wheel). Check `uv run python -c "import torch;print(torch.version.cuda, torch.cuda.is_available())"` |
| `No usable CUDA device, and the 'research' profile is not run on CPU silently` | as above | fix CUDA, or pass `--allow-cpu` deliberately |
| `CUDA out of memory at epoch …` | batch too large for the GPU | in `configs/training.yaml` halve `micro_batch` and double `grad_accum`; set `gradient_checkpointing: true`; lower `max_len` only if states fit (~480 tokens); `freeze_encoder: true` as a last resort. Start a new run (`--overwrite`) |
| OOM already at the probe with micro batch 1 | < 6 GB VRAM for a full fine-tune | `freeze_encoder: true` |
| very slow training, GPU utilisation low | CPU-bound tokenisation or AMP off | check `amp` in the hardware report; close other GPU apps; keep the project off a network drive |
| `--resume given but no saved state` | the run id differs (other dataset/seed/profile) or the state was deleted | `status` lists runs; resume needs the same profile, dataset and seed |
| `checkpoint … was made with a different configuration` | config changed since the run started | revert the change, or start a new run (`--overwrite` or another dataset name) |
| `already holds a training state` | you started without `--resume` | add `--resume`, or `--overwrite` to discard |
| Hugging Face HTTP 429 | anonymous rate limit | wait and retry (the fetch resumes); set `HF_TOKEN` |
| Binance HTTP 451 on `/fapi` | location blocked by Binance | inference falls back to cached data and marks it; the archive (data.binance.vision) is unaffected |
| Binance 429/418 during download | rate limit | handled with backoff; lower `download.concurrency` |
| `dataset 'x' already exists and is immutable` | rebuilding a dataset | new `--name`, or `--force` if you mean to replace it |
| `leakage check failed` | a split boundary, duplicate or target column problem | read `data/datasets/<name>/leakage_report.json` |
| `rows.parquet changed since it was built` | the dataset file was modified | rebuild the dataset |
| LAYA loss stuck near 1.0986 (= ln 3) and a collapse warning | too few optimizer updates or an unlucky seed (upstream #963) | more data / epochs, other seeds; this is also what an unlearnable signal looks like, so compare with the baselines |
| `predict` returns `core features missing` | not enough history (warm-up) or a gap at the end | `download` more history for the symbol |
| dashboard shows nothing | wrong working directory | run from the project root (`scripts\launch_dashboard.ps1`) |
| PowerShell refuses scripts | execution policy | `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` |

## Reproducing the final test results

1. Keep `uv.lock`, the configs and the dataset manifest (`data/datasets/<name>/manifest.json` has the
   SHA-256 of every file and the configs used).
2. Rebuild the same data: same `data.yaml` range and universe rules → `universe`, `download`,
   `dataset --name <name>`; compare the manifest hashes.
3. Re-train with the same profile and seeds; compare `fingerprint.json`.
4. `evaluate --dataset <name> --checkpoint models\best --name laya-best` and `backtest ... --split test`.

GPU kernels are not forced deterministic, so expect small numeric differences between machines.
