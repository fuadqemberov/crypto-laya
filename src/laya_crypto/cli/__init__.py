"""Command-line interface: `python -m laya_crypto.cli <command>` (or `laya-crypto <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_hardware(a):
    from ..hardware.detect import inspect, to_markdown, write_reports

    rep = inspect(".")
    j, m = write_reports(rep, "reports")
    print(to_markdown(rep))
    print(f"\nwritten: {j} and {m}")


def cmd_laya_demo(a):
    """Runs the official base checkpoint once, before any crypto training."""
    from ..datasets.serialize import QUESTION, QUESTION_ID
    from ..inference.laya_model import LayaDecisionModel
    from ..training.base_checkpoint import resolve_base

    path = resolve_base(a.base, "models/base")
    m = LayaDecisionModel(path, device=a.device)
    state = "SYMBOL=BTCUSDT\nTIMEFRAME=5m\nTREND_1H=BULLISH\nRSI=61.2\nATR_PCT=0.72\nREL_VOLUME=1.8"
    res = m.agent.predict(state, {QUESTION_ID: QUESTION})
    _print({"base_checkpoint": path, "load_seconds": round(m.load_seconds, 1), "example_state": state,
            "result": res, "note": "zero-shot base checkpoint on an illustrative state; not a trading signal"})


def cmd_universe(a):
    from ..config import load_data_config
    from ..ingestion.pipeline import build_universe

    cfg = load_data_config()
    if a.top_n:
        cfg.universe.top_n = a.top_n
    uni = build_universe(cfg, max_candidates=a.max_candidates)
    print(uni.groupby("month")["symbol"].count().tail(12).to_string())
    print(f"{uni['symbol'].nunique()} unique symbols over {uni['month'].nunique()} months")


def cmd_download(a):
    from ..config import load_data_config
    from ..ingestion.pipeline import download_market_data, selected_symbols

    cfg = load_data_config()
    if a.timeframes:
        cfg.timeframes = a.timeframes
    symbols = a.symbols or selected_symbols(cfg)
    s = download_market_data(cfg, symbols, date.fromisoformat(a.start) if a.start else None,
                             date.fromisoformat(a.end) if a.end else None)
    _print(s)


def cmd_exchange_info(a):
    from ..config import load_data_config
    from ..ingestion.live import fetch_exchange_info

    snap = fetch_exchange_info(load_data_config())
    print(f"saved metadata for {len(snap['symbols'])} symbols ({snap['note']})")


def cmd_validate(a):
    from ..config import load_data_config
    from ..ingestion import store
    from ..utils.io import write_json
    from ..validation.quality import kline_quality

    cfg = load_data_config()
    root = cfg.root_path / "processed" / "klines" / a.timeframe
    reps = []
    for p in sorted(root.glob("*.parquet")):
        df = store.load(cfg, "klines", p.stem, a.timeframe)
        reps.append(kline_quality(df, a.timeframe, p.stem))
    write_json("reports/data_quality.json", {"timeframe": a.timeframe, "symbols": reps,
                                             "coverage": store.coverage(cfg).astype(str).to_dict("records")})
    for r in reps:
        print(f"{r['symbol']:<14} rows {r['rows']:>8}  missing {r.get('missing_bars', 0):>6}  "
              f"ohlc_bad {r.get('ohlc_violations', 0):>3}  outliers {r.get('return_outliers', 0):>3}  "
              f"score {r['quality_score']:.3f}  {r.get('first_utc', '')[:10]} .. {r.get('last_utc', '')[:10]}")
    print("written: reports/data_quality.json")


def cmd_dataset(a):
    from ..datasets.builder import build_dataset

    out = build_dataset(a.name, a.symbols, force=a.force)
    print(f"dataset written to {out}")
    _print(json.loads((out / "manifest.json").read_text())["rows_by_split"])


def cmd_baselines(a):
    from ..evaluation.scoring import comparison_table
    from ..training.run_baselines import run_baselines

    run_baselines(a.dataset, a.models, a.groups, a.seed)
    print(comparison_table(a.dataset).round(4).to_string())


def cmd_ablation(a):
    from ..evaluation.ablation import run_ablation

    _print(run_ablation(a.dataset, a.model))


def cmd_train(a):
    from ..training.run import train_profile

    overrides = {}
    if a.max_train_rows:
        overrides["max_train_rows"] = a.max_train_rows
    if a.epochs:
        overrides["epochs"] = a.epochs
    if a.base:
        overrides["base"] = a.base
    if a.freeze_encoder:
        overrides["freeze_encoder"] = True
    _print(train_profile(a.profile, a.dataset, resume=a.resume, device=a.device, allow_cpu=a.allow_cpu or None,
                         seeds=a.seeds, overwrite=a.overwrite, overrides=overrides or None,
                         evaluate=not a.no_eval))


def cmd_stop(a):
    from ..config import project_path

    root = project_path("models", "checkpoints")
    runs = [root / a.run] if a.run else [p for p in root.iterdir() if (p / "status.json").exists()]
    for r in runs:
        (r / "STOP").write_text("stop requested\n")
        print(f"stop requested for {r.name}; it checkpoints and exits after the current optimizer update")


def cmd_status(a):
    from ..config import project_path
    from ..utils.io import read_json

    root = project_path("models", "checkpoints")
    for p in sorted(root.glob("*/status.json")) if root.exists() else []:
        s = read_json(p)
        print(f"{s.get('run_id'):<50} {s.get('state'):<10} epoch_done={s.get('epoch_completed')} "
              f"update={s.get('last_checkpoint_update')}")


def cmd_evaluate(a):
    from ..evaluation.report import write_comparison_report
    from ..training.run import evaluate_checkpoint

    if a.checkpoint:
        evaluate_checkpoint(a.checkpoint, a.dataset, a.name or "laya-custom", a.device, a.max_rows)
    path = write_comparison_report(a.dataset)
    print(open(path, encoding="utf-8").read())


def cmd_walk_forward(a):
    from ..evaluation.walk_forward import walk_forward_baseline

    _print(walk_forward_baseline(a.dataset, a.model, a.folds))


def cmd_hpo(a):
    from ..optimization.hpo import hpo_laya, hpo_lightgbm

    if a.target == "lightgbm":
        _print(hpo_lightgbm(a.dataset, a.trials or 30))
    else:
        _print(hpo_laya(a.dataset, a.profile, a.trials or 6, a.device, a.allow_cpu))


def cmd_backtest(a):
    from ..backtesting.run import run_backtest

    rep = run_backtest(a.dataset, a.model, a.split, min_confidence=a.min_confidence)
    _print({k: v for k, v in rep.items() if k not in ("equity_curve", "trades")})


def cmd_select_best(a):
    from ..training.run import select_best
    from ..utils.io import read_json

    run = a.run
    if not run:
        s = read_json(f"reports/{a.dataset}/laya_{a.profile}_summary.json")
        if not s:
            sys.exit("no summary for that dataset/profile; pass --run")
        run = s["best_run_by_val"]
    print(f"models/best <- {run}: {select_best(run)}")


def cmd_predict(a):
    from ..inference.predictor import Predictor

    p = Predictor(checkpoint=a.checkpoint, device=a.device)
    out = p.predict_symbols(a.symbol, a.timeframe, offline=a.offline)
    _print(out if len(out) > 1 else out[0])


def cmd_benchmark(a):
    from ..inference.predictor import Predictor

    _print(Predictor(checkpoint=a.checkpoint, device=a.device).benchmark(a.symbol, n=a.n, offline=a.offline))


def cmd_dashboard(a):
    import uvicorn

    from ..config import load_inference_config

    cfg = load_inference_config()
    host, port = a.host or cfg.host, a.port or cfg.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding to {host} exposes the unauthenticated dashboard beyond this machine")
    uvicorn.run("laya_crypto.dashboard.app:app", host=host, port=port, log_level="info")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="laya_crypto", description="LAYA crypto research trainer (no live trading)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hardware", help="inspect this machine, write reports/hardware_report.*").set_defaults(f=cmd_hardware)
    s = sub.add_parser("laya-demo", help="run the official base checkpoint once")
    s.add_argument("--base", default="english")
    s.add_argument("--device", default=None)
    s.set_defaults(f=cmd_laya_demo)

    s = sub.add_parser("universe", help="point-in-time top-N universe (delisted symbols included)")
    s.add_argument("--top-n", type=int)
    s.add_argument("--max-candidates", type=int, help="limit symbols scanned (smoke tests)")
    s.set_defaults(f=cmd_universe)

    s = sub.add_parser("download", help="download + verify + store archive data")
    s.add_argument("--symbols", nargs="*")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--timeframes", nargs="*")
    s.set_defaults(f=cmd_download)
    sub.add_parser("exchange-info", help="snapshot current contract metadata").set_defaults(f=cmd_exchange_info)

    s = sub.add_parser("validate", help="data-quality report")
    s.add_argument("--timeframe", default="5m")
    s.set_defaults(f=cmd_validate)

    s = sub.add_parser("dataset", help="features + labels + purged chronological splits")
    s.add_argument("--name", required=True)
    s.add_argument("--symbols", nargs="*")
    s.add_argument("--force", action="store_true")
    s.set_defaults(f=cmd_dataset)

    s = sub.add_parser("baselines", help="always-WAIT, trend, logistic, LightGBM, XGBoost")
    s.add_argument("--dataset", required=True)
    s.add_argument("--models", nargs="*")
    s.add_argument("--groups", nargs="*", help="feature groups to use (ablation)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(f=cmd_baselines)

    s = sub.add_parser("ablation", help="leave-one-feature-group-out study on validation")
    s.add_argument("--dataset", required=True)
    s.add_argument("--model", default="lightgbm")
    s.set_defaults(f=cmd_ablation)

    s = sub.add_parser("train", help="fine-tune LAYA with a profile")
    s.add_argument("--profile", required=True, choices=["smoke", "research", "deep"])
    s.add_argument("--dataset", required=True)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--overwrite", action="store_true", help="discard an existing run state and start again")
    s.add_argument("--device", default="auto", help="auto | cuda | cpu")
    s.add_argument("--allow-cpu", action="store_true")
    s.add_argument("--seeds", nargs="*", type=int)
    s.add_argument("--max-train-rows", type=int)
    s.add_argument("--epochs", type=int)
    s.add_argument("--base")
    s.add_argument("--freeze-encoder", action="store_true")
    s.add_argument("--no-eval", action="store_true")
    s.set_defaults(f=cmd_train)

    s = sub.add_parser("stop", help="gracefully stop a running training job")
    s.add_argument("--run")
    s.set_defaults(f=cmd_stop)
    sub.add_parser("status", help="list training runs").set_defaults(f=cmd_status)

    s = sub.add_parser("evaluate", help="comparison report (optionally score a checkpoint first)")
    s.add_argument("--dataset", required=True)
    s.add_argument("--checkpoint")
    s.add_argument("--name")
    s.add_argument("--device", default=None)
    s.add_argument("--max-rows", type=int)
    s.set_defaults(f=cmd_evaluate)

    s = sub.add_parser("walk-forward", help="purged expanding-window evaluation of a baseline")
    s.add_argument("--dataset", required=True)
    s.add_argument("--model", default="lightgbm")
    s.add_argument("--folds", type=int)
    s.set_defaults(f=cmd_walk_forward)

    s = sub.add_parser("hpo", help="bounded Optuna search on validation (lightgbm or laya)")
    s.add_argument("--dataset", required=True)
    s.add_argument("--target", choices=["lightgbm", "laya"], default="lightgbm")
    s.add_argument("--trials", type=int)
    s.add_argument("--profile", default="deep")
    s.add_argument("--device", default="auto")
    s.add_argument("--allow-cpu", action="store_true")
    s.set_defaults(f=cmd_hpo)

    s = sub.add_parser("backtest", help="event-driven portfolio backtest of a model's predictions")
    s.add_argument("--dataset", required=True)
    s.add_argument("--model", required=True, help="prediction file name under artifacts/predictions/<dataset>/")
    s.add_argument("--split", default="test")
    s.add_argument("--min-confidence", type=float)
    s.set_defaults(f=cmd_backtest)

    s = sub.add_parser("select-best", help="copy the best validated run to models/best")
    s.add_argument("--run")
    s.add_argument("--dataset")
    s.add_argument("--profile", default="research")
    s.set_defaults(f=cmd_select_best)

    for name, fn in (("predict", cmd_predict), ("benchmark", cmd_benchmark)):
        s = sub.add_parser(name)
        s.add_argument("--symbol", nargs="+", default=["BTCUSDT"])
        s.add_argument("--timeframe", default=None)
        s.add_argument("--checkpoint", default=None)
        s.add_argument("--device", default=None)
        s.add_argument("--offline", action="store_true", help="use cached archive data only (result marked stale)")
        if name == "benchmark":
            s.add_argument("--n", type=int, default=20)
        s.set_defaults(f=fn)

    s = sub.add_parser("dashboard", help="local web dashboard (127.0.0.1)")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(f=cmd_dashboard)
    return ap


def main(argv: list[str] | None = None) -> None:
    from ..utils.logging import setup_logging

    setup_logging("logs")
    a = build_parser().parse_args(argv)
    a.f(a)
