"""Resumable LAYA fine-tuning built from the official `laya.train` components.

Why not call `laya.train.finetune` directly? Two requirements it does not cover:

1. **Resume.** Upstream rewrites `checkpoint_latest/` (weights only, fp16) once per epoch; optimizer,
   scheduler, AMP scaler and RNG state are not saved, so a restart begins training from zero.
2. **Chronological calibration.** Upstream holds out a *random* calibration slice. For
   overlapping market time series that slice sits inside the training period. Here the
   temperatures are fitted on the dataset's own `calib` split, which lies after `val` in time.

Everything else is upstream code: item construction (`items_from_rows`), sequence encoding
(`encode_item`, `collate_items`), the RLCD / soft-CE objectives, the optimizer layout and cosine
schedule of `train_model`, temperature fitting (`fit_temperature_map`), the calibration report,
abstention thresholds and the checkpoint layout `laya.load` reads (`save_checkpoint`). The loop
below follows `laya.train.train_model` step for step, adding checkpoints at optimizer-update
boundaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import LEAN_CHECKPOINTS, TrainProfile, project_path
from ..utils.io import read_json, read_jsonl, sha256_file, sha256_json, write_json
from ..utils.logging import get_logger

log = get_logger("laya_trainer")

STATE_FILE = "state.pt"
STATE_META = "state.json"


class ResumeError(RuntimeError):
    pass


class DeviceError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- utilities

def resolve_device(requested: str, profile_name: str, allow_cpu: bool) -> torch.device:
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise DeviceError(
                "CUDA was requested but PyTorch cannot use a GPU "
                f"(torch {torch.__version__}, CUDA build {torch.version.cuda}). Fix: install the NVIDIA driver, then "
                "reinstall the CUDA wheel: `uv sync` (pyproject pins the cu128 index on Windows), and check "
                "`python -m laya_crypto.cli hardware`. Pass --device cpu --allow-cpu to train on CPU deliberately.")
        return torch.device("cuda", 0)
    if requested not in ("auto", "cpu"):
        return torch.device(requested)
    if profile_name != "smoke" and not allow_cpu:
        raise DeviceError(
            f"No usable CUDA device, and the '{profile_name}' profile is not run on CPU silently. "
            "Run `python -m laya_crypto.cli hardware` to see why CUDA is unavailable, or pass --allow-cpu "
            "if you really want a CPU run (expect it to be very slow).")
    return torch.device("cpu")


def gpu_stats(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    out = {"gpu_mem_allocated_gb": round(torch.cuda.memory_allocated(device) / 2**30, 3),
           "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved(device) / 2**30, 3),
           "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 3)}
    try:
        out["gpu_utilization_pct"] = torch.cuda.utilization(device)
    except Exception:  # noqa: BLE001 - needs pynvml; absence is reported, not guessed
        out["gpu_utilization_pct"] = None
    return out


def _forward(model, batch, device, amp: bool, detach_encoder: bool):
    """Same as `laya.train._forward` (kept local because upstream marks it private)."""
    args = (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
    kwargs = {"detach_encoder": detach_encoder}
    if "option_ids" in batch:
        kwargs.update(position_ids=batch["position_ids"].to(device), option_ids=batch["option_ids"].to(device))
    if amp:
        with torch.autocast(device.type, dtype=torch.float16):
            logits, _ = model(*args, **kwargs)
    else:
        logits, _ = model(*args, **kwargs)
    return logits.float()


def subsample_rows(rows: list[dict], limit: int | None) -> list[dict]:
    """Evenly spaced subset in time order (rows are chronological), deterministic."""
    if not limit or len(rows) <= limit:
        return rows
    idx = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return [rows[i] for i in idx]


def _atomic_torch_save(obj: Any, path: Path) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    digest = sha256_file(tmp)
    os.replace(tmp, path)
    return digest


# ------------------------------------------------------------------------------- trainer

class LayaRun:
    """One fine-tuning run (one profile, dataset and seed) in models/checkpoints/<run_id>/."""

    def __init__(self, run_id: str, dataset: str, profile_name: str, profile: TrainProfile, base_dir: str,
                 device: torch.device, seed: int, overrides: dict | None = None):
        self.run_id, self.dataset, self.profile_name = run_id, dataset, profile_name
        self.p, self.base_dir, self.device, self.seed = profile, base_dir, device, seed
        self.dir = project_path("models", "checkpoints", run_id)
        self.overrides = overrides or {}

    # paths
    @property
    def stop_file(self) -> Path:
        return self.dir / "STOP"

    def status(self, **kw) -> None:
        cur = read_json(self.dir / "status.json", {}) or {}
        cur.update(kw, run_id=self.run_id, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(self.dir / "status.json", cur)

    def metric(self, **kw) -> None:
        from ..utils.io import append_jsonl

        append_jsonl(self.dir / "metrics.jsonl", {"ts": time.time(), **kw})

    # fingerprints
    def fingerprint(self, dataset_manifest: dict, base_manifest: dict | None) -> dict:
        import laya

        fp = {"profile": self.p.model_dump(), "seed": self.seed, "dataset": self.dataset,
              "dataset_files": dataset_manifest["files"], "base_dir": str(self.base_dir),
              "base_revision": (base_manifest or {}).get("revision"), "laya_version": laya.__version__,
              "overrides": self.overrides}
        fp["hash"] = sha256_json({k: v for k, v in fp.items() if k != "hash"})
        return fp

    # ------------------------------------------------------------------ data
    def _items(self, tok, split: str, limit: int | None, max_len: int, head_max_len: int):
        from laya.train import items_from_rows

        path = project_path("data", "datasets", self.dataset, "laya", f"{split}.jsonl")
        if not path.exists():
            from ..config import load_data_config

            path = load_data_config().root_path / "datasets" / self.dataset / "laya" / f"{split}.jsonl"
        rows = subsample_rows(read_jsonl(path), limit)
        items, skipped = items_from_rows(tok, rows, max_len, head_max_len, label_smoothing=self.p.label_smoothing
                                         if split == "train" else 0.0)
        return rows, items, skipped

    # ------------------------------------------------------------------ resume state
    def save_state(self, model, optimizer, scheduler, scaler, order_rng: random.Random, progress: dict) -> None:
        state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict() if scaler is not None else None,
                 "order_rng": order_rng.getstate(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "progress": progress,
                 "fingerprint": self._fp["hash"]}
        path = self.dir / STATE_FILE
        if path.exists() and (self.dir / STATE_META).exists() and not LEAN_CHECKPOINTS:
            # the previous good state stays on disk (renamed, not copied) until the new one is complete
            os.replace(path, self.dir / (STATE_FILE + ".prev"))
            os.replace(self.dir / STATE_META, self.dir / (STATE_META + ".prev"))
        # lean mode keeps one state: the temp-file + rename write still never leaves it half-written
        digest = _atomic_torch_save(state, path)
        write_json(self.dir / STATE_META, {"sha256": digest, "progress": progress,
                                           "fingerprint": self._fp["hash"], "saved_utc": time.time()})

    def load_state(self) -> dict | None:
        """The newest complete, compatible state; falls back to the previous one.

        Normally `state.json` holds the SHA-256 of `state.pt`. On a synced folder (Google Drive) the
        two can arrive out of step after an abrupt shutdown: then a state that still loads completely
        and carries the same fingerprint inside it is accepted, with a warning. Truncated files fail
        to load and are skipped.
        """
        for suffix in ("", ".prev"):
            meta = read_json(self.dir / (STATE_META + suffix))
            path = self.dir / (STATE_FILE + suffix)
            if not path.exists():
                continue
            if meta and meta.get("fingerprint") not in (None, self._fp["hash"]):
                self._refuse()
            if meta and sha256_file(path) == meta["sha256"]:
                return torch.load(path, map_location="cpu", weights_only=False)
            try:
                st = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:  # noqa: BLE001 - partial / corrupt file
                log.error("%s is corrupt or partial (%r); trying the previous state", path, e)
                continue
            if st.get("fingerprint") != self._fp["hash"]:
                self._refuse()
            log.warning("%s does not match its state.json hash but is a complete state of this run "
                        "(update %s); using it", path, st["progress"]["global_update"])
            return st
        return None

    def _refuse(self):
        old = read_json(self.dir / "fingerprint.json", {})
        diff = sorted(k for k in self._fp if k != "hash" and old.get(k) != self._fp.get(k))
        raise ResumeError(f"checkpoint in {self.dir} was made with a different configuration "
                          f"(differs in: {diff}). Resume refused; start a new run instead.")

    # ------------------------------------------------------------------ main
    def train(self, resume: bool = False, overwrite: bool = False) -> dict:
        from laya.common import collate_items, uses_parallel_layout
        from laya.train import (
            draw_option_order,
            encode_item,
            load_checkpoint,
            optimizer_updates,
            rlcd_loss,
            sigma_at,
            soft_ce_loss,
        )

        from ..datasets.builder import load_dataset

        _, dmanifest = load_dataset(self.dataset)
        base_manifest = read_json(Path(self.base_dir) / "laya_base_manifest.json")
        self._fp = self.fingerprint(dmanifest, base_manifest)
        has_state = (self.dir / STATE_META).exists()
        if has_state and not resume:
            if not overwrite:
                raise ResumeError(f"{self.dir} already holds a training state. Use --resume to continue it, "
                                  "or --overwrite to discard it and start again.")
            shutil.rmtree(self.dir)
        if resume and not has_state:
            raise ResumeError(f"--resume given but no saved state in {self.dir}; nothing to resume.")
        self.dir.mkdir(parents=True, exist_ok=True)
        if not resume:  # a refused resume must never overwrite the fingerprint of the existing run
            write_json(self.dir / "fingerprint.json", self._fp)

        p, dev = self.p, self.device
        model, tok, cfg = load_checkpoint(self.base_dir)
        max_len = p.max_len or cfg.get("max_len", 512)
        head_max_len = p.head_max_len or cfg.get("head_max_len", 192)
        parallel = uses_parallel_layout(cfg)
        _, train_items, skipped = self._items(tok, "train", p.max_train_rows, max_len, head_max_len)
        _, val_items, _ = self._items(tok, "val", p.max_eval_rows, max_len, head_max_len)
        _, calib_items, _ = self._items(tok, "calib", p.max_eval_rows, max_len, head_max_len)
        if not train_items:
            raise RuntimeError(f"no LAYA training items (skipped {skipped})")
        amp = (dev.type == "cuda") if p.amp is None else bool(p.amp)
        ckpt = amp if p.gradient_checkpointing is None else bool(p.gradient_checkpointing)
        shuffle_types = ("choice",) if p.shuffle_options else ()

        if p.freeze_encoder:
            for prm in model.encoder.parameters():
                prm.requires_grad_(False)
        elif ckpt and hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = ckpt
        model.to(dev).train()
        if p.freeze_encoder:
            model.encoder.eval()

        if not resume:
            self.stop_file.unlink(missing_ok=True)
        micro, accum = p.micro_batch, p.grad_accum
        if dev.type == "cuda" and p.oom_retry and not resume:
            micro, accum = self._probe_batch(model, tok, train_items, micro, accum, max_len, head_max_len, amp,
                                             parallel, collate_items, encode_item, soft_ce_loss)
        groups = [{"params": [q for n, q in model.named_parameters() if not n.startswith("encoder.") and q.requires_grad],
                   "lr": p.head_lr}]
        if not p.freeze_encoder:
            groups.insert(0, {"params": [q for n, q in model.named_parameters()
                                         if n.startswith("encoder.") and q.requires_grad], "lr": p.encoder_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=p.weight_decay)
        total_updates = optimizer_updates(len(train_items), micro, accum, p.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda") if amp and dev.type == "cuda" else None
        params = [q for g in groups for q in g["params"]]
        steps_per_epoch = math.ceil(len(train_items) / micro)

        torch.manual_seed(self.seed)
        order_rng = random.Random(self.seed)
        progress = {"epoch": 0, "micro_step": 0, "global_update": 0, "epoch_loss_sum": 0.0, "history": [],
                    "val_history": [], "best_val_loss": None, "best_epoch": None, "bad_epochs": 0,
                    "micro_batch": micro, "grad_accum": accum, "total_updates": total_updates}
        if resume:
            st = self.load_state()
            if st is None:
                raise ResumeError(f"no valid (uncorrupted) state found in {self.dir}")
            self.stop_file.unlink(missing_ok=True)
            model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            if scaler is not None and st["scaler"] is not None:
                scaler.load_state_dict(st["scaler"])
            order_rng.setstate(st["order_rng"])
            torch.set_rng_state(st["torch_rng"])
            if st.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st["cuda_rng"])
            np.random.set_state(st["numpy_rng"])
            random.setstate(st["python_rng"])
            progress = st["progress"]
            micro, accum = progress["micro_batch"], progress["grad_accum"]
            steps_per_epoch = math.ceil(len(train_items) / micro)
            log.info("resumed %s at epoch %d, micro-step %d, optimizer update %d/%d", self.run_id,
                     progress["epoch"] + 1, progress["micro_step"], progress["global_update"], total_updates)

        info = {"train_items": len(train_items), "val_items": len(val_items), "calib_items": len(calib_items),
                "skipped": skipped, "device": str(dev), "amp": amp, "gradient_checkpointing": ckpt,
                "micro_batch": micro, "grad_accum": accum, "effective_batch": micro * accum,
                "optimizer_updates": total_updates, "max_len": max_len, "head_max_len": head_max_len}
        log.info("LAYA run %s: %s", self.run_id, json.dumps(info))
        self.status(state="running", profile=self.profile_name, dataset=self.dataset, seed=self.seed, pid=os.getpid(),
                    **info)
        cfg_out = dict(cfg, max_len=max_len, head_max_len=head_max_len)
        t_start, items_done = time.time(), 0
        stopped = False

        for epoch in range(progress["epoch"], p.epochs):
            epoch_items = list(train_items)
            random.Random(self.seed + epoch).shuffle(epoch_items)
            sigma = sigma_at(epoch, p.epochs, 0.4, 0.1)
            start_step = progress["micro_step"] if epoch == progress["epoch"] else 0
            if start_step == 0:
                progress["epoch_loss_sum"] = 0.0
            optimizer.zero_grad(set_to_none=True)
            for n_step in range(start_step, steps_per_epoch):
                start = n_step * micro
                chunk = [encode_item(tok, it, max_len, head_max_len, draw_option_order(it, order_rng, shuffle_types),
                                     parallel) for it in epoch_items[start:start + micro]]
                batch = collate_items([chunk], tok.pad_token_id)
                try:
                    logits = _forward(model, batch, dev, amp, p.freeze_encoder)
                    mask, target = batch["marker_mask"].to(dev), batch["target"].to(dev)
                    if p.loss == "rlcd":
                        loss = rlcd_loss(logits, target, mask, batch["qtype"].to(dev), sigma, 4, 0.75, 1.0)
                    else:
                        loss = soft_ce_loss(logits, target, mask)
                    window_start = (n_step // accum) * accum
                    scaled = loss / min(accum, steps_per_epoch - window_start)
                    (scaler.scale(scaled) if scaler is not None else scaled).backward()
                except torch.OutOfMemoryError as e:
                    self.status(state="failed", error="CUDA out of memory")
                    raise RuntimeError(
                        f"CUDA out of memory at epoch {epoch + 1} step {n_step}. The last checkpoint is intact. "
                        "Lower micro_batch (and raise grad_accum to keep the effective batch) in "
                        "configs/training.yaml, enable gradient_checkpointing or freeze_encoder, and start a new run. "
                        f"({e})") from e
                progress["epoch_loss_sum"] += float(loss.item())
                items_done += len(chunk)
                last_in_epoch = n_step + 1 == steps_per_epoch
                if (n_step + 1) % accum == 0 or last_in_epoch:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    progress["global_update"] += 1
                    progress["micro_step"] = n_step + 1
                    progress["epoch"] = epoch
                    g = progress["global_update"]
                    if g % p.log_every_steps == 0 or last_in_epoch:
                        elapsed = time.time() - t_start
                        rate = items_done / elapsed if elapsed > 0 else None
                        remaining_items = (total_updates - g) * micro * accum
                        self.metric(kind="train", epoch=epoch + 1, update=g, total_updates=total_updates,
                                    loss=float(loss.item()), epoch_mean_loss=progress["epoch_loss_sum"] / (n_step + 1),
                                    lr_encoder=optimizer.param_groups[0]["lr"], lr_head=optimizer.param_groups[-1]["lr"],
                                    items_per_s=round(rate, 3) if rate else None,
                                    eta_s_measured=round(remaining_items / rate) if rate else None,
                                    remaining_updates=total_updates - g, **gpu_stats(dev))
                        log.info("epoch %d/%d update %d/%d loss %.4f %s", epoch + 1, p.epochs, g, total_updates,
                                 loss.item(), f"{rate:.2f} items/s" if rate else "")
                    if self.stop_file.exists():
                        self.save_state(model, optimizer, scheduler, scaler, order_rng, progress)
                        stopped = True
                        break
                    if g % p.checkpoint_every_steps == 0 and not last_in_epoch:
                        self.save_state(model, optimizer, scheduler, scaler, order_rng, progress)
                        self.status(state="running", last_checkpoint_update=g, last_checkpoint_utc=time.time())
            if stopped:
                break
            mean = progress["epoch_loss_sum"] / steps_per_epoch
            progress["history"].append(mean)
            progress.update(epoch=epoch + 1, micro_step=0)
            if val_items and ((epoch + 1) % p.eval_every_epochs == 0 or epoch + 1 == p.epochs):
                ev = self._evaluate(model, tok, val_items, max_len, head_max_len, parallel)
                progress["val_history"].append({"epoch": epoch + 1, **ev})
                self.metric(kind="val", epoch=epoch + 1, **ev)
                log.info("epoch %d val loss %.4f acc %.4f ece %s", epoch + 1, ev["loss"], ev["accuracy"], ev.get("ece"))
                best = progress["best_val_loss"]
                if best is None or ev["loss"] < best:
                    progress.update(best_val_loss=ev["loss"], best_epoch=epoch + 1, bad_epochs=0)
                    self._save_weights(model, tok, cfg_out, self.dir / "best_uncalibrated")
                else:
                    progress["bad_epochs"] += 1
                model.train()
                if p.freeze_encoder:
                    model.encoder.eval()
            if not LEAN_CHECKPOINTS:  # the resume state already holds these weights
                self._save_weights(model, tok, cfg_out, self.dir / "checkpoint_latest")
            self.save_state(model, optimizer, scheduler, scaler, order_rng, progress)
            self.status(state="running", epoch_completed=epoch + 1, last_checkpoint_update=progress["global_update"])
            if p.early_stopping_patience and progress["bad_epochs"] >= p.early_stopping_patience:
                log.info("early stopping: no val improvement for %d epoch(s)", progress["bad_epochs"])
                break

        if stopped:
            self.status(state="stopped", stopped_at_update=progress["global_update"])
            log.info("stopped gracefully at update %d; resume with --resume", progress["global_update"])
            return {"run_id": self.run_id, "state": "stopped", "progress": progress}
        return self._finalize(model, tok, cfg_out, calib_items, val_items, max_len, head_max_len, parallel,
                              progress, info)

    def _probe_batch(self, model, tok, items, micro, accum, max_len, head_max_len, amp, parallel, collate_items,
                     encode_item, soft_ce_loss):
        """Forward+backward on the longest items; halve the micro-batch on OOM (effective batch kept)."""
        longest = sorted(items, key=lambda it: len(it["state_ids"]), reverse=True)
        while True:
            try:
                chunk = [encode_item(tok, it, max_len, head_max_len, None, parallel) for it in longest[:micro]]
                batch = collate_items([chunk], tok.pad_token_id)
                logits = _forward(model, batch, self.device, amp, self.p.freeze_encoder)
                soft_ce_loss(logits, batch["target"].to(self.device), batch["marker_mask"].to(self.device)).backward()
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                return micro, accum
            except torch.OutOfMemoryError as e:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro == 1:
                    raise RuntimeError("CUDA out of memory even at micro_batch=1: enable freeze_encoder or lower "
                                       "max_len") from e
                micro, accum = micro // 2, accum * 2
                log.warning("OOM probe: retrying with micro_batch=%d grad_accum=%d", micro, accum)

    def _evaluate(self, model, tok, items, max_len, head_max_len, parallel, temperature=None) -> dict:
        from laya.train import calibration_records, evaluate_records

        recs = calibration_records(model, tok, items, self.device, max_len, head_max_len, parallel=parallel)
        ev = evaluate_records(recs, temperature)
        return {k: ev[k] for k in ("loss", "accuracy", "ece", "brier", "mean_confidence") if k in ev}

    @staticmethod
    def _save_weights(model, tok, cfg: dict, path: Path) -> None:
        from laya.train import save_checkpoint

        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        save_checkpoint(model, tok, cfg, str(tmp))
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)

    def _finalize(self, model, tok, cfg_out, calib_items, val_items, max_len, head_max_len, parallel, progress, info):
        from laya.calibrate import fit_temperature_map
        from laya.train import (
            calibration_records,
            calibration_report,
            fit_abstention_thresholds,
            load_checkpoint,
            prior_collapse_message,
        )

        from ..datasets.serialize import QUESTION, QUESTION_ID

        best_dir = self.dir / "best_uncalibrated"
        if best_dir.exists():  # the best-validated epoch, not simply the last one
            model, _, _ = load_checkpoint(str(best_dir))
            model.to(self.device).eval()
        records = calibration_records(model, tok, calib_items, self.device, max_len, head_max_len, parallel=parallel)
        collapse = prior_collapse_message(records)
        fitted = fit_temperature_map(records)
        calib = calibration_report(records, fitted["temperature"])
        try:
            abst = fit_abstention_thresholds(records, temperature=fitted["temperature"],
                                             temperature_by_options=fitted["temperature_by_options"])
        except Exception as e:  # noqa: BLE001 - upstream raises on degenerate slices
            log.warning("abstention thresholds not fitted: %r", e)
            abst = {}
        val_cal = self._evaluate(model, tok, val_items, max_len, head_max_len, parallel, fitted["temperature"]) \
            if val_items else None
        out_cfg = dict(cfg_out, fine_tuned=True, temperature=fitted["temperature"])
        out_cfg.pop("temperature_by_options", None)  # inherited buckets would mask the new fit (upstream note)
        if fitted["temperature_by_options"]:
            out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
        out_cfg["training"] = dict(out_cfg.get("training") or {}, laya_crypto=dict(
            profile=self.profile_name, profile_config=self.p.model_dump(), dataset=self.dataset, seed=self.seed,
            fingerprint=self._fp["hash"], calibration_split="calib (chronological)", calibration=calib,
            abstention_thresholds=abst, best_epoch=progress["best_epoch"], train_info=info))
        final = self.dir / "final"
        self._save_weights(model, tok, out_cfg, final)
        write_json(final / "questions.json", {QUESTION_ID: QUESTION})
        report = {"run_id": self.run_id, "state": "completed", "profile": self.profile_name, "dataset": self.dataset,
                  "seed": self.seed, "train_info": info, "epoch_train_loss": progress["history"],
                  "val_history": progress["val_history"], "best_epoch": progress["best_epoch"],
                  "best_val_loss_uncalibrated": progress["best_val_loss"], "val_calibrated": val_cal,
                  "temperature": fitted["temperature"], "calibration": calib, "collapse_warning": collapse,
                  "final_dir": str(final), "fingerprint": self._fp["hash"]}
        write_json(self.dir / "train_report.json", report)
        if collapse:
            log.warning(collapse)
        self.status(state="completed", final_dir=str(final), best_epoch=progress["best_epoch"])
        return report


def run_id_for(profile: str, dataset: str, seed: int, tag: str = "") -> str:
    h = hashlib.sha1(f"{profile}|{dataset}|{seed}|{tag}".encode()).hexdigest()[:6]
    return f"{profile}-{dataset}-s{seed}{('-' + tag) if tag else ''}-{h}"


def profile_with_hardware(p: TrainProfile, hw_profile: dict | None) -> TrainProfile:
    """Fit micro-batch / AMP / checkpointing to the measured GPU, keeping the effective batch."""
    if not hw_profile or hw_profile.get("device") != "cuda":
        return p
    eff = p.micro_batch * p.grad_accum
    micro = min(p.micro_batch, hw_profile["micro_batch"])
    upd = {"micro_batch": micro, "grad_accum": max(1, eff // micro)}
    if p.amp is None:
        upd["amp"] = hw_profile["amp"]
    if p.gradient_checkpointing is None:
        upd["gradient_checkpointing"] = hw_profile["gradient_checkpointing"]
    return p.model_copy(update=upd)

