"""Structured logging: human-readable console lines plus a JSON-lines file under logs/."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("laya_crypto")
    root.setLevel(level)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "laya_crypto.jsonl", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
    except OSError:
        root.warning("could not open log file in %s; console logging only", log_dir)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger("laya_crypto." + name)


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"fields": fields})
