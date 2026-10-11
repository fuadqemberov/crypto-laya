"""Inference output schema (tiny real LAYA checkpoint, offline cached data) and dashboard endpoints."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import synthetic_klines
from laya_crypto.config import DataConfig
from laya_crypto.ingestion import store
from laya_crypto.utils.io import write_parquet


@pytest.fixture
def cached_data(tmp_path, monkeypatch):
    cfg = DataConfig(root=str(tmp_path / "data"))
    now_ms = (pd.Timestamp.now(tz="UTC").floor("5min").value // 10**6) - 2600 * 300_000
    for i, s in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT")):
        write_parquet(store.processed_path(cfg, "klines", s, "5m"), synthetic_klines(2600, seed=i, start=now_ms))
    import laya_crypto.inference.predictor as pred

    monkeypatch.setattr(pred, "load_data_config", lambda: cfg)
    return cfg


def test_predict_schema_offline(cached_data, tiny_ckpt):
    from laya_crypto.inference.predictor import Predictor

    out = Predictor(checkpoint=str(tiny_ckpt), device="cpu").predict_one("SOLUSDT", "5m", offline=True)
    assert "error" not in out, out
    assert set(out["probabilities"]) == {"LONG", "SHORT", "WAIT"}
    assert abs(sum(out["probabilities"].values()) - 1) < 0.01
    assert out["decision"] in ("LONG", "SHORT", "WAIT") and isinstance(out["abstained"], bool)
    assert out["data_source"] == "archive" and "latency_ms" in out and out["model_version"]["checkpoint"]
    assert out["data_age_minutes"] >= 0


def test_predict_missing_data_is_explicit_error(cached_data, tiny_ckpt):
    from laya_crypto.inference.predictor import Predictor

    out = Predictor(checkpoint=str(tiny_ckpt), device="cpu").predict_one("NOPEUSDT", "5m", offline=True)
    assert "error" in out and "decision" not in out


def test_predict_without_checkpoint_errors(cached_data, tmp_path):
    from laya_crypto.inference.predictor import InferenceError, Predictor

    p = Predictor(checkpoint=str(tmp_path / "none"))
    with pytest.raises(InferenceError):
        _ = p.model


def test_dashboard_endpoints(monkeypatch, tmp_path):
    from laya_crypto.dashboard import app as appmod
    from laya_crypto.dashboard import jobs

    client = TestClient(appmod.app)
    assert client.get("/").status_code == 200
    for path in ("/api/hardware", "/api/runs", "/api/datasets", "/api/data", "/api/logs", "/api/symbols", "/api/job"):
        r = client.get(path)
        assert r.status_code == 200, (path, r.text)
    assert client.get("/api/runs/../../etc/metrics").status_code == 404
    assert client.post("/api/train", json={"profile": "bogus", "dataset": "x"}).status_code == 400
    monkeypatch.setattr(jobs, "current", lambda: {"alive": True, "pid": 1, "cmd": ["python", "-m", "x"]})
    r = client.post("/api/train", json={"profile": "smoke", "dataset": "x"})
    assert r.status_code == 409  # no concurrent training jobs
