"""Local dashboard API (FastAPI). Every endpoint reads real project files or runs real operations."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import load_data_config, project_path
from ..utils.io import read_json, read_jsonl
from . import jobs


def _clean(o):
    """NaN/inf are not valid JSON: send them as null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(_clean(content), allow_nan=False, default=str).encode("utf-8")


app = FastAPI(title="LAYA crypto trainer", docs_url="/api/docs", default_response_class=SafeJSONResponse)
STATIC = Path(__file__).parent / "static"
_predictor_lock = threading.Lock()
_predictors: dict[str, object] = {}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# ----------------------------------------------------------------------------- hardware
@app.get("/api/hardware")
def hardware():
    rep = read_json(project_path("reports", "hardware_report.json"))
    vm = psutil.virtual_memory()
    live = {"cpu_percent": psutil.cpu_percent(interval=0.2), "ram_used_gb": round((vm.total - vm.available) / 2**30, 2),
            "ram_total_gb": round(vm.total / 2**30, 2)}
    try:
        from ..hardware.detect import nvidia_smi

        live["nvidia"] = nvidia_smi()
    except Exception as e:  # noqa: BLE001
        live["nvidia"] = {"available": False, "reason": repr(e)}
    return {"report": rep, "live": live}


@app.post("/api/hardware/refresh")
def hardware_refresh():
    from ..hardware.detect import inspect, write_reports

    rep = inspect(project_path())
    write_reports(rep, project_path("reports"))
    return rep


# --------------------------------------------------------------------------------- data
@app.get("/api/data")
def data():
    cfg = load_data_config()
    from ..ingestion import store

    cov = store.coverage(cfg)
    summary = []
    if not cov.empty:
        for (kind, tf), g in cov.groupby(["kind", "timeframe"], dropna=False):
            summary.append({"kind": kind, "timeframe": tf, "symbols": int(g["symbol"].nunique()),
                            "rows": int(g["rows"].sum()), "first": str(g["first_utc"].min()),
                            "last": str(g["last_utc"].max())})
    return {"coverage_summary": summary,
            "coverage": cov.astype(str).to_dict("records") if not cov.empty else [],
            "quality": read_json(project_path("reports", "data_quality.json")),
            "universe": read_json(cfg.root_path / "universe" / "universe_report.json"),
            "download": read_json(cfg.root_path / "download_report.json")}


@app.get("/api/datasets")
def datasets():
    root = load_data_config().root_path / "datasets"
    out = []
    for m in sorted(root.glob("*/manifest.json")) if root.exists() else []:
        man = read_json(m)
        out.append({k: man.get(k) for k in ("name", "created_utc", "timeframe", "rows", "rows_by_split",
                                            "label_distribution", "realised_decision_time_ms")}
                   | {"symbols": len(man.get("symbols", [])), "features": len(man.get("feature_columns", []))})
    return out


# ----------------------------------------------------------------------------- training
@app.get("/api/runs")
def runs():
    root = project_path("models", "checkpoints")
    out = []
    for st in sorted(root.glob("*/status.json")) if root.exists() else []:
        s = read_json(st) or {}
        s["has_final"] = (st.parent / "final" / "rl_agent_config.json").exists()
        s["has_resume_state"] = (st.parent / "state.json").exists()
        rep = read_json(st.parent / "train_report.json")
        if rep:
            s["evaluation"] = rep.get("evaluation")
            s["val_calibrated"] = rep.get("val_calibrated")
        out.append(s)
    best = read_json(project_path("models", "best", "selection.json"))
    return {"runs": out, "best": best, "job": jobs.current()}


@app.get("/api/runs/{run_id}/metrics")
def run_metrics(run_id: str):
    d = project_path("models", "checkpoints", run_id)
    if not d.is_dir() or d.parent != project_path("models", "checkpoints"):
        raise HTTPException(404, "unknown run")
    return {"metrics": read_jsonl(d / "metrics.jsonl"), "status": read_json(d / "status.json"),
            "report": read_json(d / "train_report.json")}


class TrainRequest(BaseModel):
    profile: str = "smoke"
    dataset: str
    resume: bool = False
    device: str = "auto"
    allow_cpu: bool = False
    overwrite: bool = False


@app.post("/api/train")
def train(req: TrainRequest):
    if req.profile not in ("smoke", "research", "deep"):
        raise HTTPException(400, "unknown profile")
    try:
        return jobs.start_training(req.profile, req.dataset, req.resume, req.device, req.allow_cpu, req.overwrite)
    except jobs.JobConflict as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/train/stop")
def stop():
    return {"stop_requested": jobs.request_stop()}


@app.get("/api/job")
def job():
    j = jobs.current()
    return {"job": j, "log_tail": jobs.tail(j["log"]) if j else ""}


class SelectRequest(BaseModel):
    run_id: str


@app.post("/api/select-best")
def select(req: SelectRequest):
    from ..training.run import select_best

    try:
        return {"best": select_best(req.run_id)}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


# -------------------------------------------------------------------------- experiments
@app.get("/api/experiments/{dataset}")
def experiments(dataset: str):
    from ..evaluation.scoring import comparison_table

    tab = comparison_table(dataset)
    summaries = {p.stem: read_json(p) for p in project_path("reports", dataset).glob("laya_*_summary.json")}
    extra = {n: read_json(project_path("reports", dataset, f"{n}.json"))
             for n in ("walk_forward_lightgbm", "ablation_lightgbm", "hpo_lightgbm")}
    reliability = {}
    for p in project_path("reports", dataset).glob("*.json"):
        rep = read_json(p)
        if isinstance(rep, dict) and "splits" in rep and "val" in rep["splits"]:
            reliability[rep["model"]] = rep["splits"]["val"]["classification"]["reliability"]
    return {"table": tab.where(pd.notna(tab), None).to_dict("records"), "laya_summaries": summaries,
            "extra": extra, "reliability_val": reliability}


@app.get("/api/backtests/{dataset}")
def backtests(dataset: str):
    d = project_path("reports", dataset, "backtests")
    return [read_json(p) for p in sorted(d.glob("*.json"))] if d.exists() else []


@app.get("/api/backtests/{dataset}/{name}")
def backtest_detail(dataset: str, name: str):
    d = project_path("reports", dataset, "backtests")
    summary = read_json(d / f"{name}.json")
    if summary is None:
        raise HTTPException(404, "unknown backtest")
    eq = pd.read_parquet(d / f"{name}_equity.parquet")
    step = max(1, len(eq) // 600)
    tr_path = d / f"{name}_trades.parquet"
    trades = pd.read_parquet(tr_path) if tr_path.exists() else pd.DataFrame()
    return {"summary": summary, "equity": eq.iloc[::step].to_dict("records"),
            "trades": trades.tail(300).where(pd.notna(trades.tail(300)), None).to_dict("records")}


@app.post("/api/backtests/{dataset}/{model}/run")
def backtest_run(dataset: str, model: str, split: str = "test"):
    from ..backtesting.run import run_backtest

    try:
        r = run_backtest(dataset, model, split)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return {k: v for k, v in r.items() if k not in ("equity_curve", "trades")}


# ----------------------------------------------------------------------------- inference
class PredictRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str | None = None
    offline: bool = False
    checkpoint: str | None = None


@app.post("/api/predict")
def predict(req: PredictRequest):
    from ..inference.predictor import InferenceError, Predictor

    key = req.checkpoint or "default"
    with _predictor_lock:
        if key not in _predictors:
            _predictors[key] = Predictor(checkpoint=req.checkpoint)
        p = _predictors[key]
        try:
            return p.predict_one(req.symbol.upper(), req.timeframe, req.offline)
        except InferenceError as e:
            raise HTTPException(400, str(e)) from e


@app.get("/api/symbols")
def symbols():
    cfg = load_data_config()
    root = cfg.root_path / "processed" / "klines" / cfg.base_timeframe
    return sorted(p.stem for p in root.glob("*.parquet")) if root.exists() else []


# ---------------------------------------------------------------------------------- logs
@app.get("/api/logs")
def logs():
    root = project_path("logs")
    return sorted((p.name for p in root.glob("*.log")), reverse=True) if root.exists() else []


@app.get("/api/logs/{name}")
def log_file(name: str, lines: int = 200):
    p = project_path("logs", name)
    if p.parent != project_path("logs") or not p.exists():
        raise HTTPException(404, "unknown log")
    return {"name": name, "tail": jobs.tail(p, lines)}


@app.get("/api/reports/{dataset}/comparison")
def comparison(dataset: str):
    from ..evaluation.report import write_comparison_report

    return {"markdown": write_comparison_report(dataset).read_text(encoding="utf-8")}
