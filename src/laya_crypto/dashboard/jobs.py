"""One training job at a time, run as a real `python -m laya_crypto.cli train ...` subprocess."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from ..config import project_path
from ..utils.io import read_json, write_json


class JobConflict(RuntimeError):
    pass


def _state_path() -> Path:
    return project_path("artifacts", "jobs", "current.json")


def current() -> dict | None:
    job = read_json(_state_path())
    if not job:
        return None
    alive = False
    try:
        p = psutil.Process(job["pid"])
        alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE and "laya_crypto" in " ".join(p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        alive = False
    job["alive"] = alive
    if not alive and job.get("state") == "running":
        job["state"] = "exited"
        job["exit_code"] = job.get("exit_code")
        write_json(_state_path(), job)
    return job


def start_training(profile: str, dataset: str, resume: bool = False, device: str = "auto",
                   allow_cpu: bool = False, overwrite: bool = False) -> dict:
    job = current()
    if job and job["alive"]:
        raise JobConflict(f"a training job is already running (pid {job['pid']}): {' '.join(job['cmd'])}")
    cmd = [sys.executable, "-m", "laya_crypto.cli", "train", "--profile", profile, "--dataset", dataset,
           "--device", device]
    if resume:
        cmd.append("--resume")
    if allow_cpu:
        cmd.append("--allow-cpu")
    if overwrite:
        cmd.append("--overwrite")
    log = project_path("logs", f"train_{profile}_{dataset}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=str(project_path()), stdout=fh, stderr=subprocess.STDOUT,
                                creationflags=flags, start_new_session=os.name != "nt")
    job = {"pid": proc.pid, "cmd": cmd, "log": str(log), "started_utc": time.time(), "state": "running",
           "profile": profile, "dataset": dataset}
    write_json(_state_path(), job)
    return job


def request_stop() -> list[str]:
    """Graceful stop: every running run writes a checkpoint at its next optimizer update and exits."""
    root = project_path("models", "checkpoints")
    stopped = []
    for st in root.glob("*/status.json") if root.exists() else []:
        s = read_json(st) or {}
        if s.get("state") == "running":
            (st.parent / "STOP").write_text("stop requested from dashboard\n")
            stopped.append(st.parent.name)
    return stopped


def tail(path: str | Path, lines: int = 80) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except FileNotFoundError:
        return ""
