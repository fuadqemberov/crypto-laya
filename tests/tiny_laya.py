"""A tiny, randomly initialised LAYA checkpoint (real tokenizer, 2-layer ModernBERT) for fast tests.

It exercises the real LAYA code paths (build_model, items, losses, save/load, laya.load) without
the 842 MB base weights. Its predictions are meaningless by design.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "models" / "base" / "english"


def tokenizer_available() -> bool:
    return (BASE / "tokenizer" / "tokenizer.json").exists() and (BASE / "encoder" / "config.json").exists()


def make_tiny_checkpoint(dest: Path, seed: int = 0) -> Path:
    import torch
    from laya.common import build_model
    from laya.train import save_checkpoint
    from transformers import AutoTokenizer

    dest = Path(dest)
    enc_dir = dest / "_enc_cfg"
    enc_dir.mkdir(parents=True, exist_ok=True)
    ecfg = json.loads((BASE / "encoder" / "config.json").read_text())
    ecfg.update(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                global_attn_every_n_layers=1, local_attention=64)
    (enc_dir / "config.json").write_text(json.dumps(ecfg))
    cfg = json.loads((BASE / "rl_agent_config.json").read_text())
    cfg.update(max_len=512, head_max_len=128)
    cfg.pop("temperature_by_options", None)
    torch.manual_seed(seed)
    model = build_model(cfg, encoder_dir=str(enc_dir))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and ("norm" in name or name.endswith(".weight") and p.numel() == 64):
                p.fill_(1.0)
            elif name.endswith(".bias"):
                p.zero_()
            else:
                p.normal_(0, 0.02)
    tok = AutoTokenizer.from_pretrained(str(BASE / "tokenizer"))
    save_checkpoint(model.float(), tok, cfg, str(dest))
    shutil.rmtree(enc_dir)
    (dest / "laya_base_manifest.json").write_text(json.dumps({"name": "tiny-test", "revision": "test"}))
    return dest
