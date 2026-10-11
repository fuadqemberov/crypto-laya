"""Profile-level orchestration: seeds, evaluation through the official inference path, model selection."""
from __future__ import annotations

import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from ..config import load_inference_config, load_training_config, project_path
from ..datasets.builder import load_dataset
from ..evaluation.scoring import save_predictions, score_model
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger
from .base_checkpoint import resolve_base
from .laya_trainer import LayaRun, profile_with_hardware, resolve_device, run_id_for

log = get_logger("train")


def _hardware_profile() -> dict | None:
    rep = read_json(project_path("reports", "hardware_report.json"))
    return rep.get("profile") if rep else None


def evaluate_checkpoint(final_dir: str, dataset: str, model_name: str, device: str,
                        max_rows: int | None = None, splits=("val", "test")) -> dict:
    """Score a checkpoint on val (selection) and test (final report) through `laya.load`."""
    from ..inference.laya_model import LayaDecisionModel

    df, _ = load_dataset(dataset)
    rows = df[df["split"].isin(splits)]
    if max_rows:
        parts = []
        for s in splits:
            part = rows[rows["split"] == s]
            if len(part) > max_rows:
                idx = np.linspace(0, len(part) - 1, max_rows).round().astype(int)
                part = part.iloc[idx]
            parts.append(part)
        import pandas as pd

        rows = pd.concat(parts)
    rows = rows.reset_index(drop=True)
    model = LayaDecisionModel(final_dir, device=device)
    t = time.perf_counter()
    proba = model.predict_proba(rows["state"].tolist(), batch_size=32)
    secs = time.perf_counter() - t
    save_predictions(dataset, model_name, rows, proba)
    info = {"checkpoint": final_dir, "rows_scored": len(rows), "inference_seconds": round(secs, 1),
            "model_only_ms_per_row": round(1000 * secs / max(1, len(rows)), 2), **model.version}
    return score_model(dataset, model_name, rows, proba, info)


def train_profile(profile_name: str, dataset: str, resume: bool = False, device: str = "auto",
                  allow_cpu: bool | None = None, seeds: list[int] | None = None, overwrite: bool = False,
                  overrides: dict | None = None, evaluate: bool = True, tag: str = "") -> dict:
    tcfg = load_training_config()
    if profile_name not in tcfg.profiles:
        raise KeyError(f"unknown profile {profile_name!r}; configs/training.yaml defines {sorted(tcfg.profiles)}")
    profile = tcfg.profiles[profile_name]
    if overrides:
        profile = profile.model_copy(update=overrides)
    profile = profile_with_hardware(profile, _hardware_profile())
    dev = resolve_device(device, profile_name, tcfg.allow_cpu if allow_cpu is None else allow_cpu)
    base_dir = resolve_base(profile.base, project_path("models", "base"))
    results = []
    for seed in seeds or profile.seeds:
        rid = run_id_for(profile_name, dataset, seed, tag)
        run = LayaRun(rid, dataset, profile_name, profile, base_dir, dev, seed, overrides)
        rep = run.train(resume=resume, overwrite=overwrite)
        if rep["state"] != "completed":
            return {"state": rep["state"], "run_id": rid}
        if evaluate:
            ev = evaluate_checkpoint(rep["final_dir"], dataset, f"laya-{rid}", str(dev), profile.max_eval_rows)
            rep["evaluation"] = {s: {"accuracy": v["classification"]["accuracy"],
                                     "log_loss": v["classification"]["log_loss"], "ece": v["classification"]["ece"],
                                     "expectancy": v["signals"]["expectancy"]} for s, v in ev["splits"].items()}
        results.append(rep)
    summary = summarize(profile_name, dataset, results)
    return summary


def summarize(profile_name: str, dataset: str, results: list[dict]) -> dict:
    """Mean/stdev over seeds; the best seed is chosen on VALIDATION log loss only."""
    def val_metric(r):
        ev = r.get("evaluation", {}).get("val")
        return ev["log_loss"] if ev else r.get("best_val_loss_uncalibrated") or float("inf")

    best = min(results, key=val_metric)
    agg = {}
    for split in ("val", "test"):
        vals = [r["evaluation"][split] for r in results if split in r.get("evaluation", {})]
        if vals:
            agg[split] = {k: {"mean": statistics.fmean(v[k] for v in vals if v[k] is not None)
                              if any(v[k] is not None for v in vals) else None,
                              "stdev": statistics.pstdev([v[k] for v in vals if v[k] is not None])
                              if sum(v[k] is not None for v in vals) > 1 else None}
                          for k in ("accuracy", "log_loss", "ece", "expectancy")}
    summary = {"profile": profile_name, "dataset": dataset, "seeds": [r["seed"] for r in results],
               "runs": [r["run_id"] for r in results], "best_run_by_val": best["run_id"],
               "aggregate_over_seeds": agg, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(project_path("reports", dataset, f"laya_{profile_name}_summary.json"), summary)
    return summary


def select_best(run_id: str) -> str:
    """Copy a completed run's final checkpoint to models/best (what inference loads by default)."""
    src = project_path("models", "checkpoints", run_id, "final")
    if not (src / "rl_agent_config.json").exists():
        raise FileNotFoundError(f"{src} has no completed checkpoint")
    ck = Path(load_inference_config().checkpoint)
    dst = ck if ck.is_absolute() else project_path(str(ck))
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    write_json(tmp / "selection.json", {"run_id": run_id, "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                         time.gmtime()),
                                        "rule": "lowest validation log loss among seeds / runs"})
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return str(dst)
