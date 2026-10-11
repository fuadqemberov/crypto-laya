"""Backtest a model's stored predictions on one split under identical conditions for every model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_backtest_config, load_data_config, load_labeling_config, project_path
from ..datasets.builder import load_dataset
from ..evaluation.scoring import PROB_COLS, predictions_dir
from ..ingestion import store
from ..ingestion.live import load_exchange_info
from ..utils.io import write_json, write_parquet
from .engine import Backtester, make_bars, performance


def run_backtest(dataset: str, model: str, split: str = "test", min_confidence: float | None = None) -> dict:
    cfg, bcfg, lcfg = load_data_config(), load_backtest_config(), load_labeling_config()
    if min_confidence is not None:
        bcfg.min_confidence = min_confidence
    rows, manifest = load_dataset(dataset)
    tf = manifest["timeframe"]
    preds = pd.read_parquet(predictions_dir(dataset) / f"{model}.parquet")
    preds = preds[preds["split"] == split]
    if preds.empty:
        raise ValueError(f"{model} has no predictions on split {split!r}")
    meta = rows[["symbol", "open_time", "regime_trend", *(["regime_abnormal"] if "regime_abnormal" in rows else [])]]
    sig = preds.merge(meta, on=["symbol", "open_time"], how="left")
    sig["confidence"] = sig[PROB_COLS].max(axis=1)
    bars = {}
    for sym in sig["symbol"].unique():
        frame = pd.read_parquet(cfg.root_path / "features" / tf / f"{sym}.parquet",
                                columns=["open_time", "open", "high", "low", "close", "atr_pct"])
        bars[sym] = make_bars(frame, store.load(cfg, "funding_rate", sym), tf)
    bt = Backtester(bars, bcfg, lcfg, tf, load_exchange_info(cfg))
    res = bt.run(sig)
    perf = performance(res, bcfg.initial_equity_usdt)
    perf.update(model=model, dataset=dataset, split=split, config=bcfg.model_dump(),
                exchange_filters="applied" if load_exchange_info(cfg) else
                "not applied (no exchange-info snapshot; run `exchange-info`)",
                signals=int((sig["pred"] != "WAIT").sum()), decisions=int(len(sig)))
    out = project_path("reports", dataset, "backtests")
    write_parquet(out / f"{model}_{split}_trades.parquet", res["trades"] if len(res["trades"]) else
                  pd.DataFrame(columns=["symbol"]))
    write_parquet(out / f"{model}_{split}_equity.parquet", res["equity"])
    write_json(out / f"{model}_{split}.json", perf)
    perf["equity_curve"] = res["equity"]
    perf["trades"] = res["trades"]
    return perf


def backtest_all(dataset: str, split: str = "test") -> pd.DataFrame:
    rows = []
    for p in sorted(predictions_dir(dataset).glob("*.parquet")):
        try:
            r = run_backtest(dataset, p.stem, split)
        except ValueError:
            continue
        rows.append({k: r.get(k) for k in ("model", "trades", "return_pct", "max_drawdown_pct", "profit_factor",
                                           "win_rate", "sharpe_daily_annualised", "fees_usdt", "funding_usdt")})
    return pd.DataFrame(rows).replace({np.nan: None})
