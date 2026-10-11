"""Filesystem helpers: atomic writes, hashing and JSON that never leaves half-written files."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode())


def _atomic_target(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Write via a temp file in the same directory, fsync, then rename over the target."""
    path = Path(path)
    fd, tmp = _atomic_target(path)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    path = Path(path)
    fd, tmp = _atomic_target(path)
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
                n += 1
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return n


def append_jsonl(path: str | os.PathLike, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def write_parquet(path: str | os.PathLike, df: pd.DataFrame) -> None:
    path = Path(path)
    fd, tmp = _atomic_target(path)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
