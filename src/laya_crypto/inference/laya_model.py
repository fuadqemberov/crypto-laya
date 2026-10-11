"""The single LAYA inference path used by evaluation, backtests, the CLI and the dashboard.

Probabilities come from the official `laya.load(...).predict_batch(...)`, with the temperatures the
fine-tuning run fitted on the chronological calibration split.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..datasets.serialize import QUESTION, QUESTION_ID
from ..labeling.triple_barrier import CLASSES


class LayaDecisionModel:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        import laya

        self.checkpoint = Path(checkpoint)
        if not (self.checkpoint / "rl_agent_config.json").exists():
            raise FileNotFoundError(f"{self.checkpoint} is not a LAYA checkpoint (no rl_agent_config.json)")
        from ..utils.device import default_device

        t = time.perf_counter()
        self.device = device or default_device()  # never a visible-but-unsupported GPU
        self.agent = laya.load(str(self.checkpoint), device=self.device)
        self.load_seconds = time.perf_counter() - t
        qpath = self.checkpoint / "questions.json"
        self.questions = json.loads(qpath.read_text()) if qpath.exists() else {QUESTION_ID: QUESTION}
        cfg = json.loads((self.checkpoint / "rl_agent_config.json").read_text())
        meta = (cfg.get("training") or {}).get("laya_crypto") or {}
        self.version = {"checkpoint": str(self.checkpoint), "fingerprint": meta.get("fingerprint"),
                        "profile": meta.get("profile"), "dataset": meta.get("dataset"), "seed": meta.get("seed"),
                        "fine_tuned": bool(cfg.get("fine_tuned")), "temperature": cfg.get("temperature")}

    def predict_proba(self, states: list[str], batch_size: int = 32) -> np.ndarray:
        if not states:
            return np.zeros((0, len(CLASSES)))
        res = self.agent.predict_batch(states, self.questions, batch_size=batch_size)
        out = np.zeros((len(states), len(CLASSES)))
        for i, r in enumerate(res):
            probs = r["answers"][QUESTION_ID]["probabilities"]
            out[i] = [float(probs[c]) for c in CLASSES]
        s = out.sum(1, keepdims=True)
        if not np.all(np.isfinite(out)) or np.any(np.abs(s - 1) > 0.02):
            raise ValueError("LAYA returned an invalid probability distribution")
        return out / s
