"""Local Parquet store: raw zips -> data/processed/<kind>/<tf>/<SYMBOL>.parquet, plus a manifest."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import DataConfig
from ..utils.io import read_json, sha256_file, write_json, write_parquet
from ..utils.logging import get_logger
from .parsers import PARSERS, TIME_KEY, normalise

log = get_logger("store")


def processed_path(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> Path:
    return cfg.root_path / "processed" / kind / (timeframe or "_") / f"{symbol}.parquet"


def raw_dir(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> Path:
    return cfg.root_path / "raw" / kind / (timeframe or "_") / symbol


def load(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> pd.DataFrame | None:
    p = processed_path(cfg, kind, symbol, timeframe)
    return pd.read_parquet(p) if p.exists() else None


def build_processed(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> dict | None:
    """(Re)build one processed table from every verified zip on disk. Idempotent."""
    zips = sorted(raw_dir(cfg, kind, symbol, timeframe).glob("*.zip"))
    if not zips:
        return None
    parser = PARSERS[kind]
    frames, bad = [], []
    for z in zips:
        try:
            frames.append(parser(z))
        except Exception as e:  # noqa: BLE001 - a corrupt file is reported, not fatal
            bad.append({"file": z.name, "error": repr(e)})
            log.error("parse failed %s: %r", z, e)
    if not frames:
        return {"kind": kind, "symbol": symbol, "timeframe": timeframe, "rows": 0, "parse_errors": bad}
    df, dropped = normalise(pd.concat(frames, ignore_index=True), kind)
    out = processed_path(cfg, kind, symbol, timeframe)
    write_parquet(out, df)
    key = TIME_KEY[kind]
    rec = {"kind": kind, "symbol": symbol, "timeframe": timeframe, "rows": len(df),
           "duplicates_dropped": int(dropped), "first_ms": int(df[key].iloc[0]), "last_ms": int(df[key].iloc[-1]),
           "source_files": len(zips), "parse_errors": bad, "path": str(out.relative_to(cfg.root_path)),
           "sha256": sha256_file(out), "source": f"{cfg.archive_base_url} ({kind})",
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    update_manifest(cfg, rec)
    return rec


def manifest_path(cfg: DataConfig) -> Path:
    return cfg.root_path / "manifest.json"


def update_manifest(cfg: DataConfig, rec: dict) -> None:
    m = read_json(manifest_path(cfg), {"tables": {}})
    m["tables"][f"{rec['kind']}/{rec['timeframe'] or '_'}/{rec['symbol']}"] = rec
    m["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(manifest_path(cfg), m)


def read_manifest(cfg: DataConfig) -> dict:
    return read_json(manifest_path(cfg), {"tables": {}})


def coverage(cfg: DataConfig) -> pd.DataFrame:
    tables = read_manifest(cfg)["tables"].values()
    df = pd.DataFrame(list(tables))
    if df.empty:
        return df
    df["first_utc"] = pd.to_datetime(df["first_ms"], unit="ms", utc=True)
    df["last_utc"] = pd.to_datetime(df["last_ms"], unit="ms", utc=True)
    return df[["kind", "timeframe", "symbol", "rows", "first_utc", "last_utc", "source_files", "duplicates_dropped"]]
