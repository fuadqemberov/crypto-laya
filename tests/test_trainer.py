"""LAYA fine-tuning engine on a tiny real LAYA model: training, checkpoint, interruption, resume."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from laya_crypto.config import TrainProfile
from laya_crypto.datasets.serialize import laya_row, serialize_row
from laya_crypto.training import laya_trainer
from laya_crypto.training.laya_trainer import LayaRun, ResumeError, resolve_device
from laya_crypto.utils.io import write_jsonl


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(laya_trainer, "project_path", lambda *p: tmp_path.joinpath(*p))
    import laya_crypto.datasets.builder as builder

    monkeypatch.setattr(builder, "load_dataset", lambda name, cfg=None: (None, {"files": {"rows.parquet": "x"}}))
    rng = np.random.default_rng(0)
    for split, n in (("train", 12), ("val", 6), ("calib", 6)):
        rows = []
        for i in range(n):
            lab = ["LONG", "SHORT", "WAIT"][i % 3]
            state = serialize_row({"rsi": float(rng.uniform(20, 80)), "adx": float(rng.uniform(10, 40)),
                                   "regime_trend": "SIDEWAYS", "regime_vol": "NORMAL_VOL"}, "BTCUSDT", "5m")
            rows.append(laya_row(state, lab))
        write_jsonl(tmp_path / "data" / "datasets" / "ds" / "laya" / f"{split}.jsonl", rows)
    return tmp_path


def profile(**kw):
    base = dict(epochs=2, micro_batch=2, grad_accum=2, max_len=512, head_max_len=128, checkpoint_every_steps=1,
                log_every_steps=1, max_train_rows=None, max_eval_rows=None, seeds=[0])
    base.update(kw)
    return TrainProfile(**base)


def weights(path):
    from safetensors.torch import load_file

    return load_file(str(path / "model.safetensors"))


def test_train_save_reload(project, tiny_ckpt):
    run = LayaRun("full", "ds", "smoke", profile(), str(tiny_ckpt), torch.device("cpu"), 0)
    rep = run.train()
    assert rep["state"] == "completed" and len(rep["epoch_train_loss"]) == 2
    final = project / "models" / "checkpoints" / "full" / "final"
    import laya

    agent = laya.load(str(final), device="cpu")
    q = json.loads((final / "questions.json").read_text())
    probs = agent.predict("SYMBOL=BTCUSDT\nRSI=50", q)["answers"]["decision"]["probabilities"]
    assert set(probs) == {"LONG", "SHORT", "WAIT"} and abs(sum(probs.values()) - 1) < 0.02
    cfg = json.loads((final / "rl_agent_config.json").read_text())
    assert cfg["fine_tuned"] and cfg["training"]["laya_crypto"]["calibration_split"].startswith("calib")
    metrics = [json.loads(x) for x in (final.parent / "metrics.jsonl").read_text().splitlines()]
    assert {"loss", "lr_encoder", "lr_head", "items_per_s", "remaining_updates"} <= set(metrics[0])


def test_interrupt_and_resume_matches_uninterrupted(project, tiny_ckpt, monkeypatch):
    dev = torch.device("cpu")
    LayaRun("ref", "ds", "smoke", profile(), str(tiny_ckpt), dev, 0).train()

    run = LayaRun("int", "ds", "smoke", profile(), str(tiny_ckpt), dev, 0)
    orig_metric = LayaRun.metric

    def stop_at_4(self, **kw):  # request a graceful stop mid-epoch-2 (update 4 of 6)
        orig_metric(self, **kw)
        if kw.get("kind") == "train" and kw.get("update") == 4:
            self.stop_file.write_text("stop")
    monkeypatch.setattr(LayaRun, "metric", stop_at_4)
    first = run.train()
    assert first["state"] == "stopped" and first["progress"]["global_update"] == 4
    monkeypatch.setattr(LayaRun, "metric", orig_metric)

    with pytest.raises(ResumeError):  # never silently restart from zero
        LayaRun("int", "ds", "smoke", profile(), str(tiny_ckpt), dev, 0).train(resume=False)
    rep = LayaRun("int", "ds", "smoke", profile(), str(tiny_ckpt), dev, 0).train(resume=True)
    assert rep["state"] == "completed"
    a = weights(project / "models" / "checkpoints" / "ref" / "final")
    b = weights(project / "models" / "checkpoints" / "int" / "final")
    assert a.keys() == b.keys()
    worst = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    assert worst < 1e-3, f"resumed weights differ by {worst}"
    ref_hist = json.loads((project / "models/checkpoints/ref/train_report.json").read_text())["epoch_train_loss"]
    assert np.allclose(ref_hist, rep["epoch_train_loss"], atol=1e-4)


def test_resume_refuses_changed_config_and_handles_corruption(project, tiny_ckpt, monkeypatch):
    dev = torch.device("cpu")
    run = LayaRun("x", "ds", "smoke", profile(), str(tiny_ckpt), dev, 0)
    orig_metric = LayaRun.metric

    def stop_at_2(self, **kw):
        orig_metric(self, **kw)
        if kw.get("kind") == "train" and kw.get("update") == 2:
            self.stop_file.write_text("stop")
    monkeypatch.setattr(LayaRun, "metric", stop_at_2)
    run.train()
    monkeypatch.setattr(LayaRun, "metric", orig_metric)
    with pytest.raises(ResumeError, match="different configuration"):
        LayaRun("x", "ds", "smoke", profile(head_lr=5e-4), str(tiny_ckpt), dev, 0).train(resume=True)
    d = project / "models" / "checkpoints" / "x"
    assert (d / "state.pt.prev").exists()
    (d / "state.pt").write_bytes(b"truncated")  # simulate a partial write of the newest state
    st = LayaRun("x", "ds", "smoke", profile(), str(tiny_ckpt), dev, 0)
    st._fp = json.loads((d / "fingerprint.json").read_text())
    loaded = st.load_state()
    assert loaded is not None and loaded["progress"]["global_update"] == 1  # fell back to the previous state


def test_full_profile_refuses_silent_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(laya_trainer.DeviceError):
        resolve_device("auto", "research", allow_cpu=False)
    with pytest.raises(laya_trainer.DeviceError):
        resolve_device("cuda", "smoke", allow_cpu=True)
    assert resolve_device("auto", "smoke", allow_cpu=False).type == "cpu"
    assert resolve_device("cpu", "deep", allow_cpu=True).type == "cpu"


def test_hardware_profile_keeps_effective_batch():
    from laya_crypto.training.laya_trainer import profile_with_hardware

    p = profile(micro_batch=8, grad_accum=8)
    out = profile_with_hardware(p, {"device": "cuda", "micro_batch": 2, "amp": True, "gradient_checkpointing": True})
    assert out.micro_batch * out.grad_accum == 64 and out.micro_batch == 2 and out.amp is True


def test_state_out_of_step_with_meta_is_accepted_if_complete(project, tiny_ckpt, monkeypatch):
    """Synced folders can deliver a new state.json with an older (complete) state.pt."""
    dev = torch.device("cpu")
    orig_metric = LayaRun.metric

    def stop_at_2(self, **kw):
        orig_metric(self, **kw)
        if kw.get("kind") == "train" and kw.get("update") == 2:
            self.stop_file.write_text("stop")
    monkeypatch.setattr(LayaRun, "metric", stop_at_2)
    LayaRun("y", "ds", "smoke", profile(), str(tiny_ckpt), dev, 0).train()
    monkeypatch.setattr(LayaRun, "metric", orig_metric)
    d = project / "models" / "checkpoints" / "y"
    meta = json.loads((d / "state.json").read_text())
    meta["sha256"] = "0" * 64  # meta from a "newer" save than the file that arrived
    (d / "state.json").write_text(json.dumps(meta))
    st = LayaRun("y", "ds", "smoke", profile(), str(tiny_ckpt), dev, 0)
    st._fp = json.loads((d / "fingerprint.json").read_text())
    assert st.load_state()["progress"]["global_update"] == 2
