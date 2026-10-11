# Running on Google Colab

Notebook: [`notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb). Open it in Colab with
*File → Open notebook → GitHub*, repository `fuadqemberov/crypto-laya`, branch `master`.
Then *Runtime → Change runtime type → GPU* and *Runtime → Run all*.

## Where things live

| location | content | survives a disconnect |
|---|---|---|
| `STORAGE` on Google Drive (default `MyDrive/laya-crypto-storage`) | `data/` (processed tables, universe, datasets, features, labels), `models/` (base checkpoint, runs with resume state, `best/`), `reports/`, `artifacts/` (predictions, baseline models), `logs/`, `configs/` | **yes** |
| `/content/laya-crypto-trainer` | code (cloned from GitHub) | no; re-cloned by the notebook |
| `/content/raw_cache` | downloaded archive zips | no; not needed again |

This is controlled by environment variables the notebook sets; they work the same on any machine:

| variable | effect |
|---|---|
| `LAYA_CRYPTO_STORAGE` | root of `data/`, `models/`, `reports/`, `artifacts/`, `logs/` |
| `LAYA_CRYPTO_RAW_DIR` | raw zip cache (disposable) |
| `LAYA_CRYPTO_CONFIG_DIR` | configs folder (the notebook copies the repo's configs to Drive once) |
| `LAYA_CRYPTO_LEAN=1` | lean checkpoints, see below |
| `HF_TOKEN` | Hugging Face token, read from the Colab secret `HF_TOKEN` if you add one |

## Why nothing is lost or repeated

- **Downloads are incremental.** Each processed Parquet table records, in its own metadata (written
  in the same atomic write as the data), which archive zips it contains. A new session plans only
  zips not yet in the table, so the deleted raw cache is never fetched again. Verified: after
  deleting the cache, the same `download` planned 0 files; extending the date range fetched only
  the 7 new files.
- **Datasets are immutable** and reused by name.
- **Training re-runs itself correctly.** `train ... --auto-resume` skips seeds that already
  completed with the same configuration fingerprint (and reuses their stored evaluation), resumes a
  seed that has a saved state from its last optimizer update, and starts the rest.
- **Writes are atomic** (temp file + rename). On a synced Drive folder, `state.json` and `state.pt`
  can arrive out of step after an abrupt shutdown; the trainer then accepts a state that loads
  completely and carries the run's fingerprint inside it, and rejects truncated files.

What can still be lost: Drive uploads files in the background. If the VM is killed within minutes
of writing a 4.8 GB state, Drive may still hold the previous one, so you lose at most the updates
since the previous save. Run the last notebook cell (`drive.flush_and_unmount()`) before closing the
tab on purpose.

## Drive space

| item | size |
|---|---|
| base checkpoint | 0.85 GB |
| processed data, top 100, from 2024 at 5m with metrics | several GB |
| one run, `LAYA_CRYPTO_LEAN=1` | ~7 GB (one 4.8 GB resume state + `best_uncalibrated/` + `final/`) |
| one run, normal | ~12 GB (adds the previous state and `checkpoint_latest/`) |

The free 15 GB Drive fits the data and roughly one lean run. Research and deep profiles train three
seeds, so use a larger Drive plan, or delete `models/checkpoints/<run>/state.pt` of completed seeds
(their `final/` checkpoint is what evaluation and inference use).

## Colab specifics

- GPU: a T4 (15 GB) is detected as `medium_vram`: micro batch 4 × accumulation 16, AMP and gradient
  checkpointing on. Full fine-tuning fits. No speed is promised; each run logs its measured items/s
  and ETA to `metrics.jsonl`.
- Install: the notebook uses `pip install -e .` so Colab's preinstalled CUDA PyTorch is kept
  (`uv sync` would select the CPU wheel on Linux).
- Session limits: free sessions end after at most ~12 hours, and earlier when idle. Re-run the
  notebook; the training cell continues where it stopped.
- **Binance live API**: `fapi.binance.com` returns HTTP 451 to US-hosted machines, including Colab.
  The historical archive is unaffected, so download, training, evaluation and backtests work. Run
  live `predict` on your own PC.

## Using the results on your PC

Sync the Drive folder with *Google Drive for desktop* and point the project at it:

```powershell
$env:LAYA_CRYPTO_STORAGE = "G:\My Drive\laya-crypto-storage"
uv run python -m laya_crypto.cli dashboard
uv run python -m laya_crypto.cli predict --symbol BTCUSDT
```
