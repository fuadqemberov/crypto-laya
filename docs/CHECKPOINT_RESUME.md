# Checkpoints and resume

## What upstream LAYA provides

`laya.train.finetune` rewrites `checkpoint_latest/` after every epoch with fp16 weights only.
Optimizer moments, the LR schedule position, the AMP scaler and RNG states are not saved, so a
crash means restarting from zero (or from those weights with a fresh optimizer, which is not the
same run).

## What this project adds

`training/laya_trainer.py` saves a full resume state at optimizer-update boundaries (every
`checkpoint_every_steps` updates, at every epoch end, and when a stop is requested):

- model weights in **fp32** (the exact training values, not the fp16 export);
- AdamW state, cosine scheduler state, `GradScaler` state;
- RNG states: torch CPU, torch CUDA, NumPy, Python, and the option-shuffle generator;
- progress: epoch, micro-step inside the epoch, global update, running epoch loss, loss history,
  validation history, best validation loss and epoch, early-stopping counter, the effective
  micro-batch / accumulation (after any OOM adjustment);
- the run fingerprint.

The epoch's item order is a pure function of `seed + epoch`, so a resumed run skips exactly the
micro-batches already consumed. Data-loader workers are not used (items are encoded in the training
loop), so there is no sampler state to lose.

### Integrity

1. The previous good state is renamed to `state.pt.prev` / `state.json.prev`.
2. The new state is written to `state.pt.tmp`, fsynced, hashed, and atomically renamed to `state.pt`.
3. `state.json` records its SHA-256 and the fingerprint.

On resume, a state whose hash does not match (partial or corrupt file) is rejected and the previous
one is used; if both are bad, resume refuses with an error. LAYA-format checkpoint folders are
written to `<name>.tmp` and renamed, so `final/` is never half-written.

### Compatibility

`--resume` compares the stored fingerprint with the current one (profile settings, seed, dataset file
hashes, base checkpoint revision, LAYA version) and refuses on any difference, listing the
differing keys. Starting a run whose directory already holds a state without `--resume` is also
refused (use `--overwrite` to discard it). Training never silently restarts from zero.

### Tested

`tests/test_trainer.py` (on a tiny real LAYA model) verifies that a run stopped after 4 of 6 updates
and resumed produces the same weights (max abs difference < 1e-3 in the fp16 export) and the same
epoch losses as an uninterrupted run; that a changed config is refused; that a truncated `state.pt`
falls back to `state.pt.prev`; and that resume without saved state is an error.

### Lean mode and synced folders

`LAYA_CRYPTO_LEAN=1` keeps a single resume state (no `.prev`) and skips the per-epoch
`checkpoint_latest/` folder, which the resume state makes redundant: about 6 GB less per run. The
write stays atomic, but the fallback to a previous state is gone.

On a synced folder (Google Drive), `state.json` may describe a newer save than the `state.pt` that
arrived. A state that loads completely and carries this run's fingerprint inside it is then
accepted with a warning. A truncated file is rejected.

`train --auto-resume` makes the command re-runnable: completed seeds (same fingerprint) are skipped,
saved states are resumed, and the remaining seeds start. See [COLAB.md](COLAB.md).

## Usage

```powershell
# stop gracefully (writes STOP; the run saves at the next optimizer update and exits)
uv run python -m laya_crypto.cli stop
# or press Ctrl+C: the last periodic state (at most checkpoint_every_steps updates old) is used
uv run python -m laya_crypto.cli train --profile deep --dataset main --resume
```

## Limitations

- Ctrl+C / power loss loses the updates since the last periodic save (`checkpoint_every_steps`).
- A mid-run CUDA OOM stops the run with instructions; the last saved state stays valid but cannot
  be resumed with a different micro batch (the fingerprint changes). Start a new run with lower
  `micro_batch` / higher `grad_accum`.
- Bitwise-identical resumption on GPU is not guaranteed (cuDNN/cuBLAS kernels are not forced
  deterministic); the optimizer trajectory is the same up to floating-point noise.
- Disk: a state for the 421 M-parameter model is ~4.8 GB, and the previous state is kept as well.
