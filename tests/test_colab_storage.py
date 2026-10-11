"""Persistent storage root, incremental processed tables (raw cache is disposable), auto-resume."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from conftest import synthetic_klines
from laya_crypto.config import DataConfig
from laya_crypto.ingestion import store
from laya_crypto.ingestion.parsers import KLINE_COLS


def test_storage_root_env(tmp_path):
    code = ("from laya_crypto.config import project_path, load_data_config, PROJECT_ROOT;"
            "print(project_path('models','x'));print(project_path('configs'));"
            "print(load_data_config().root_path);print(load_data_config().raw_path)")
    env = dict(os.environ, LAYA_CRYPTO_STORAGE=str(tmp_path / "drive"), LAYA_CRYPTO_RAW_DIR=str(tmp_path / "raw"))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    models, configs, data, raw = out.stdout.split()
    assert models == str(tmp_path / "drive" / "models" / "x")
    assert not configs.startswith(str(tmp_path))  # code/configs stay in the project
    assert data == str(tmp_path / "drive" / "data") and raw == str(tmp_path / "raw")


def _write_zip(cfg, kind, symbol, tf, name, df):
    d = store.raw_dir(cfg, kind, symbol, tf)
    d.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.assign(ignore=0)[KLINE_COLS].to_csv(buf, index=False)
    with zipfile.ZipFile(d / name, "w") as z:
        z.writestr(name.replace(".zip", ".csv"), buf.getvalue())


def test_incremental_tables_survive_deleted_raw_cache(tmp_path):
    cfg = DataConfig(root=str(tmp_path / "data"))
    df = synthetic_klines(300)
    _write_zip(cfg, "klines", "X", "5m", "X-5m-a.zip", df.iloc[:100])
    _write_zip(cfg, "klines", "X", "5m", "X-5m-b.zip", df.iloc[100:200])
    assert store.build_processed(cfg, "klines", "X", "5m")["rows"] == 200
    import shutil

    shutil.rmtree(cfg.raw_path)  # e.g. a new Colab session: local cache gone, Drive table kept
    _write_zip(cfg, "klines", "X", "5m", "X-5m-c.zip", df.iloc[200:])
    rec = store.build_processed(cfg, "klines", "X", "5m")
    assert rec["rows"] == 300 and rec["new_files"] == 1 and rec["source_files"] == 3
    assert store.incorporated_sources(cfg, "klines", "X", "5m") == {"X-5m-a.zip", "X-5m-b.zip", "X-5m-c.zip"}
    again = store.build_processed(cfg, "klines", "X", "5m")
    assert again["new_files"] == 0 and again["rows"] == 300
    out = store.load(cfg, "klines", "X", "5m")
    pd.testing.assert_series_equal(out["close"], df["close"].reset_index(drop=True), check_names=False)


def test_superseding_file_replaces_its_range_without_double_count(tmp_path):
    cfg = DataConfig(root=str(tmp_path / "data"))
    flow = pd.DataFrame({"open_time": [0, 60_000, 120_000], "buy_qty": [1.0, 1, 1], "sell_qty": [1.0, 1, 1],
                         "buy_quote": [1.0, 1, 1], "sell_quote": [1.0, 1, 1], "trades": [1, 1, 1]})
    p = store.processed_path(cfg, "agg_trades", "X")
    store._write_with_sources(p, flow.iloc[:2], {"X-aggTrades-2025-08-01.zip"})  # daily file, earlier session

    import laya_crypto.ingestion.store as st

    monthly = flow.assign(buy_qty=[1.0, 1, 1])
    raw = store.raw_dir(cfg, "agg_trades", "X")
    raw.mkdir(parents=True)
    (raw / "X-aggTrades-2025-08.zip").write_bytes(b"")
    orig = st.PARSERS["agg_trades"]
    st.PARSERS["agg_trades"] = lambda path: monthly
    try:
        store.build_processed(cfg, "agg_trades", "X")
    finally:
        st.PARSERS["agg_trades"] = orig
    out = store.load(cfg, "agg_trades", "X")
    assert out["buy_qty"].tolist() == [1.0, 1.0, 1.0]  # replaced, not summed to 2.0


@pytest.mark.usefixtures("tiny_ckpt")
def test_auto_resume_skips_completed_and_resumes_interrupted(tmp_path, monkeypatch, tiny_ckpt):
    import torch

    from laya_crypto.config import TrainingConfig, TrainProfile
    from laya_crypto.datasets.serialize import laya_row, serialize_row
    from laya_crypto.training import laya_trainer, run
    from laya_crypto.utils.io import write_jsonl

    monkeypatch.setattr(laya_trainer, "project_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(run, "project_path", lambda *p: tmp_path.joinpath(*p))
    import laya_crypto.datasets.builder as builder

    monkeypatch.setattr(builder, "load_dataset", lambda name, cfg=None: (None, {"files": {"rows.parquet": "x"}}))
    for split in ("train", "val", "calib"):
        rows = [laya_row(serialize_row({"rsi": 30.0 + i}, "BTCUSDT", "5m"), ["LONG", "SHORT", "WAIT"][i % 3])
                for i in range(6)]
        write_jsonl(tmp_path / "data" / "datasets" / "ds" / "laya" / f"{split}.jsonl", rows)
    prof = TrainProfile(epochs=2, micro_batch=2, grad_accum=1, max_len=512, head_max_len=128, max_train_rows=None,
                        max_eval_rows=None, seeds=[0, 1], checkpoint_every_steps=1, log_every_steps=1)
    monkeypatch.setattr(run, "load_training_config", lambda: TrainingConfig(profiles={"smoke": prof}))
    monkeypatch.setattr(run, "resolve_base", lambda base, root: str(tiny_ckpt))
    monkeypatch.setattr(run, "_hardware_profile", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    calls = []
    orig_train = laya_trainer.LayaRun.train

    def counting(self, resume=False, overwrite=False):
        calls.append((self.seed, resume))
        if self.seed == 1 and not resume:  # interrupt seed 1 after its 2nd update (simulated disconnect)
            orig_metric = laya_trainer.LayaRun.metric

            def stop(this, **kw):
                orig_metric(this, **kw)
                if kw.get("kind") == "train" and kw.get("update") == 2:
                    this.stop_file.write_text("stop")
            monkeypatch.setattr(laya_trainer.LayaRun, "metric", stop)
            try:
                return orig_train(self, resume, overwrite)
            finally:
                monkeypatch.setattr(laya_trainer.LayaRun, "metric", orig_metric)
        return orig_train(self, resume, overwrite)
    monkeypatch.setattr(laya_trainer.LayaRun, "train", counting)

    first = run.train_profile("smoke", "ds", evaluate=False, auto_resume=True)
    assert first["state"] == "stopped" and calls == [(0, False), (1, False)]
    second = run.train_profile("smoke", "ds", evaluate=False, auto_resume=True)
    assert calls[2:] == [(1, True)]  # seed 0 not retrained, seed 1 resumed
    assert len(second["runs"]) == 2
    rep1 = json.loads((laya_trainer.LayaRun("x", "ds", "smoke", prof, str(tiny_ckpt), torch.device("cpu"), 1)
                       .dir.parent / second["runs"][1] / "train_report.json").read_text())
    assert rep1["state"] == "completed"
