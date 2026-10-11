"""Build an immutable, leakage-checked dataset from processed market data.

data/datasets/<name>/
  rows.parquet          features, labels, targets, split, serialized state
  manifest.json         configs, boundaries, counts, feature list, SHA-256 of every file
  leakage_report.json
  laya/{train,val,calib,test}.jsonl   LAYA `{state, questions, expected|gold}` rows
  laya/{split}.csv      the same as text/label CSV (laya-train --data ... --text-column state)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    DataConfig,
    DatasetConfig,
    FeaturesConfig,
    LabelingConfig,
    load_data_config,
    load_dataset_config,
    load_features_config,
    load_labeling_config,
)
from ..features.pipeline import AuxData, build_features
from ..features.regime import GMMRegime
from ..ingestion import store
from ..ingestion.universe import membership_mask, universe_path
from ..labeling.triple_barrier import CLASSES, label_frame, soft_label_distribution
from ..utils.io import read_json, sha256_file, write_json, write_jsonl, write_parquet
from ..utils.logging import get_logger
from ..utils.timeframes import tf_ms
from . import leakage
from .serialize import laya_row, serialize_row
from .splits import SPLITS, assign_splits, boundaries

log = get_logger("dataset")

TARGET_PREFIXES = ("long_", "short_", "fwd_ret", "mfe", "mae", "p_", "tp_pct", "sl_pct", "entry_price")


def load_aux(cfg: DataConfig, symbol: str) -> AuxData:
    tf = cfg.base_timeframe
    return AuxData(funding=store.load(cfg, "funding_rate", symbol), metrics=store.load(cfg, "metrics", symbol),
                   premium=store.load(cfg, "premium_index_klines", symbol, tf),
                   mark=store.load(cfg, "mark_price_klines", symbol, tf),
                   index=store.load(cfg, "index_price_klines", symbol, tf),
                   flow=store.load(cfg, "agg_trades", symbol), depth=store.load(cfg, "book_depth", symbol),
                   metrics_lag_ms=cfg.metrics_publication_lag_ms)


def load_context(cfg: DataConfig) -> dict[str, pd.DataFrame]:
    out = {}
    for s in cfg.context_symbols:
        df = store.load(cfg, "klines", s, cfg.base_timeframe)
        if df is not None:
            out[s] = df[["open_time", "close"]]
    return out


def symbol_rows(cfg: DataConfig, fcfg: FeaturesConfig, lcfg: LabelingConfig, dcfg: DatasetConfig, symbol: str,
                context: dict[str, pd.DataFrame], universe: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    tf = dcfg.timeframe
    base = store.load(cfg, "klines", symbol, tf)
    if base is None or len(base) < 500:
        log.warning("%s: not enough %s candles (%s)", symbol, tf, 0 if base is None else len(base))
        return pd.DataFrame(), {}
    aux = load_aux(cfg, symbol)
    ff = build_features(base, fcfg, tf, aux, context)
    frame = ff.frame
    one_min = store.load(cfg, "klines", symbol, "1m") if lcfg.ambiguity == "resolve_1m" else None
    labels = label_frame(frame, frame["atr_pct"], lcfg, tf, aux.funding, one_min)
    if lcfg.soft_targets:
        labels = labels.join(soft_label_distribution(frame, frame["atr_pct"], lcfg, tf, aux.funding))
    feats_dir = cfg.root_path / "features" / tf
    write_parquet(feats_dir / f"{symbol}.parquet", frame)
    write_parquet(cfg.root_path / "labels" / tf / f"{symbol}.parquet",
                  labels.assign(open_time=frame["open_time"].to_numpy()))
    rows = pd.concat([frame, labels.drop(columns=[c for c in labels.columns if c in frame.columns])], axis=1)
    keep = rows["bar_present"].astype(bool) & rows["label"].notna() & rows["atr_pct"].notna()
    if universe is not None and len(universe):
        keep &= membership_mask(universe, symbol, rows["open_time"]).to_numpy()
    if dcfg.sample_every > 1:
        keep &= ((rows["open_time"] // tf_ms(tf)) % dcfg.sample_every == 0).to_numpy()
    rows = rows[keep]
    if dcfg.max_rows_per_symbol:
        rows = rows.tail(dcfg.max_rows_per_symbol)
    rows = rows.assign(symbol=symbol, bar_ms=tf_ms(tf))
    return rows.reset_index(drop=True), ff.groups


def build_dataset(name: str, symbols: list[str] | None = None, force: bool = False,
                  cfg: DataConfig | None = None, fcfg: FeaturesConfig | None = None,
                  lcfg: LabelingConfig | None = None, dcfg: DatasetConfig | None = None) -> Path:
    cfg, fcfg = cfg or load_data_config(), fcfg or load_features_config()
    lcfg, dcfg = lcfg or load_labeling_config(), dcfg or load_dataset_config()
    out = cfg.root_path / "datasets" / name
    if (out / "manifest.json").exists() and not force:
        raise FileExistsError(f"dataset {name!r} already exists and is immutable; choose a new name or --force")
    uni_p = universe_path(cfg)
    universe = pd.read_parquet(uni_p) if uni_p.exists() else None
    if symbols is None:
        if universe is None:
            raise FileNotFoundError("no universe: run the `universe` command or pass --symbols")
        symbols = sorted(universe["symbol"].unique())
    elif universe is not None:
        log.info("explicit symbol list given; rows still restricted to point-in-time universe months")
    context = load_context(cfg)
    parts, groups = [], {}
    for i, s in enumerate(symbols, 1):
        rows, g = symbol_rows(cfg, fcfg, lcfg, dcfg, s, context, universe)
        if len(rows):
            parts.append(rows)
            for k, cols in g.items():  # union over symbols, first-seen order
                groups[k] = list(dict.fromkeys(groups.get(k, []) + cols))
        log.info("[%d/%d] %s: %d rows", i, len(symbols), s, len(rows))
    if not parts:
        raise RuntimeError("no rows produced; check downloads and the universe")
    df = pd.concat(parts, ignore_index=True).sort_values(["decision_time", "symbol"]).reset_index(drop=True)
    feature_cols = [c for cols in groups.values() for c in cols]
    for c in feature_cols:
        if c not in df:
            df[c] = np.nan  # a symbol lacking a data type has the column all-missing, never invented
    embargo_bars = dcfg.split.embargo_bars if dcfg.split.embargo_bars is not None else lcfg.max_hold_bars
    embargo_ms = embargo_bars * tf_ms(dcfg.timeframe)
    bounds = boundaries(df["decision_time"], dcfg.split)
    df["split"] = assign_splits(df, dcfg.split, embargo_ms)
    dropped = int(df["split"].isna().sum())
    df = df[df["split"].notna()].reset_index(drop=True)
    if dcfg.wait_downsample < 1.0:
        rng = np.random.default_rng(0)
        drop = (df["split"] == "train") & (df["label"] == "WAIT") & (rng.random(len(df)) > dcfg.wait_downsample)
        df = df[~drop].reset_index(drop=True)
    # unsupervised regime detector fitted on TRAIN rows only, then applied to every row
    try:
        gmm = GMMRegime().fit(df[df["split"] == "train"])
        df["regime_gmm"] = gmm.predict(df).astype(float)
        feature_cols = [*feature_cols, "regime_gmm"]
        groups.setdefault("regime", []).append("regime_gmm")
    except (KeyError, ValueError) as e:
        log.warning("GMM regime skipped: %r", e)
    df["state"] = [serialize_row(r, r["symbol"], dcfg.timeframe) for r in df.to_dict("records")]
    rep = leakage.check(df, feature_cols)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "leakage_report.json", rep)
    if not rep["ok"]:
        raise RuntimeError(f"leakage check failed, see {out / 'leakage_report.json'}")
    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIXES)]
    keep_cols = ["symbol", "open_time", "decision_time", "label_end_time", "bar_ms", "split", "label", "state",
                 "open", "high", "low", "close", "volume", "regime_trend", "regime_vol",
                 *feature_cols, *[c for c in target_cols if c not in feature_cols]]
    df = df[list(dict.fromkeys(keep_cols))]
    write_parquet(out / "rows.parquet", df)
    files = {"rows.parquet": sha256_file(out / "rows.parquet")}
    for s in SPLITS:
        part = df[df["split"] == s]
        if lcfg.soft_targets and "p_LONG" in part:
            rows = (laya_row(st, soft={c: float(part.at[i, f"p_{c}"]) for c in CLASSES})
                    for i, st in zip(part.index, part["state"], strict=True))
        else:
            rows = (laya_row(st, lab) for st, lab in zip(part["state"], part["label"], strict=True))
        write_jsonl(out / "laya" / f"{s}.jsonl", rows)
        part[["state", "label"]].to_csv(out / "laya" / f"{s}.csv", index=False)
        files[f"laya/{s}.jsonl"] = sha256_file(out / "laya" / f"{s}.jsonl")
    manifest = {
        "name": name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timeframe": dcfg.timeframe, "symbols": sorted(df["symbol"].unique().tolist()),
        "rows": len(df), "rows_by_split": rep["rows_by_split"], "dropped_by_purge_embargo": dropped,
        "boundaries_ms": bounds, "realised_decision_time_ms": _realised_bounds(df),
        "embargo_ms": embargo_ms, "label_distribution": rep["label_dist_by_split"],
        "feature_columns": feature_cols, "feature_groups": groups, "target_columns": target_cols, "files": files,
        "configs": {"data": cfg.model_dump(), "features": fcfg.model_dump(), "labeling": lcfg.model_dump(),
                    "dataset": dcfg.model_dump()},
        "universe_file_sha256": sha256_file(uni_p) if uni_p.exists() else None,
        "data_manifest_sha256": sha256_file(store.manifest_path(cfg)) if store.manifest_path(cfg).exists() else None,
    }
    write_json(out / "manifest.json", manifest)
    log.info("dataset %s: %d rows %s", name, len(df), rep["rows_by_split"])
    return out


def _realised_bounds(df: pd.DataFrame) -> dict[str, list[int]]:
    g = df.groupby("split")["decision_time"]
    return {s: [int(g.min()[s]), int(g.max()[s])] for s in SPLITS if s in g.groups}


def load_dataset(name: str, cfg: DataConfig | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or load_data_config()
    d = cfg.root_path / "datasets" / name
    manifest = read_json(d / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"dataset {name!r} not found under {d}")
    actual = sha256_file(d / "rows.parquet")
    if actual != manifest["files"]["rows.parquet"]:
        raise RuntimeError(f"dataset {name!r} rows.parquet changed since it was built (hash mismatch)")
    return pd.read_parquet(d / "rows.parquet"), manifest
