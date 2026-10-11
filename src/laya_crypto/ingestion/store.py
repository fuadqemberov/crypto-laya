"""Local Parquet store: raw zips -> data/processed/<kind>/<tf>/<SYMBOL>.parquet, plus a manifest."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..config import DataConfig
from ..utils.io import read_json, sha256_file, write_json
from ..utils.logging import get_logger
from .parsers import PARSERS, TIME_KEY, normalise

log = get_logger("store")


def processed_path(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> Path:
    return cfg.root_path / "processed" / kind / (timeframe or "_") / f"{symbol}.parquet"


def raw_dir(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> Path:
    return cfg.raw_path / kind / (timeframe or "_") / symbol


def load(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> pd.DataFrame | None:
    p = processed_path(cfg, kind, symbol, timeframe)
    return pd.read_parquet(p) if p.exists() else None


SOURCES_KEY = b"laya_crypto_sources"


def incorporated_sources(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> set[str]:
    """Archive zip names already merged into a processed table (stored in its Parquet metadata)."""
    import pyarrow.parquet as pq

    p = processed_path(cfg, kind, symbol, timeframe)
    if not p.exists():
        return set()
    meta = pq.read_schema(p).metadata or {}
    raw = meta.get(SOURCES_KEY)
    return set(json.loads(raw)) if raw else set()


def _write_with_sources(path: Path, df: pd.DataFrame, sources: set[str]) -> None:
    import os
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[SOURCES_KEY] = json.dumps(sorted(sources)).encode()
    table = table.replace_schema_metadata(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)  # data and its source list change together, atomically
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def build_processed(cfg: DataConfig, kind: str, symbol: str, timeframe: str | None = None) -> dict | None:
    """Merge newly downloaded zips into the processed table. Incremental and idempotent.

    Zips already recorded in the table are skipped, so the raw download cache can be deleted (or
    live on a disposable disk) without losing data. A new zip replaces the existing rows inside its
    own time range, so a monthly archive that supersedes daily files of the same month is not
    double counted.
    """
    out = processed_path(cfg, kind, symbol, timeframe)
    key = TIME_KEY[kind]
    done = incorporated_sources(cfg, kind, symbol, timeframe)
    zips = sorted(z for z in raw_dir(cfg, kind, symbol, timeframe).glob("*.zip") if z.name not in done)
    existing = pd.read_parquet(out) if out.exists() else None
    if not zips and existing is None:
        return None
    parser = PARSERS[kind]
    bad, parsed_names = [], set()
    df = existing
    dropped = 0
    for z in zips:
        try:
            part = parser(z)
        except Exception as e:  # noqa: BLE001 - a corrupt file is reported, not fatal
            bad.append({"file": z.name, "error": repr(e)})
            log.error("parse failed %s: %r", z, e)
            continue
        parsed_names.add(z.name)
        if part.empty:
            continue
        if df is not None and len(df):
            lo, hi = part[key].min(), part[key].max()
            df = df[(df[key] < lo) | (df[key] > hi)]
        df = part if df is None else pd.concat([df, part], ignore_index=True)
    if df is None or df.empty:
        return {"kind": kind, "symbol": symbol, "timeframe": timeframe, "rows": 0, "parse_errors": bad}
    if parsed_names or existing is None:
        df, dropped = normalise(df, kind)
        _write_with_sources(out, df, done | parsed_names)
    sources = done | parsed_names
    rec = {"kind": kind, "symbol": symbol, "timeframe": timeframe, "rows": len(df),
           "duplicates_dropped": int(dropped), "first_ms": int(df[key].iloc[0]), "last_ms": int(df[key].iloc[-1]),
           "source_files": len(sources), "new_files": len(parsed_names), "parse_errors": bad,
           "path": str(out.relative_to(cfg.root_path)), "sha256": sha256_file(out),
           "source": f"{cfg.archive_base_url} ({kind})",
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if parsed_names or existing is None:
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
