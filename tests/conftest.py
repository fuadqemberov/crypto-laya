from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from tiny_laya import make_tiny_checkpoint, tokenizer_available  # noqa: E402

T0 = 1_735_689_600_000  # 2025-01-01 00:00 UTC


def synthetic_klines(n: int = 3000, seed: int = 0, step_ms: int = 300_000, start: int = T0,
                     drift: float = 0.0, vol: float = 0.002, price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = price * np.exp(np.cumsum(r))
    open_ = np.r_[price, close[:-1]]
    spread = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol_ = rng.lognormal(3, 0.5, n)
    t = start + np.arange(n, dtype="int64") * step_ms
    return pd.DataFrame({"open_time": t, "open": open_, "high": high, "low": low, "close": close, "volume": vol_,
                         "close_time": t + step_ms - 1, "quote_volume": vol_ * close, "count": rng.integers(10, 100, n).astype(float),
                         "taker_buy_volume": vol_ * rng.uniform(0.3, 0.7, n),
                         "taker_buy_quote_volume": vol_ * close * 0.5})


@pytest.fixture
def klines():
    return synthetic_klines()


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory):
    if not tokenizer_available():
        pytest.skip("LAYA tokenizer not present (run `python -m laya_crypto.cli laya-demo` once)")
    return make_tiny_checkpoint(tmp_path_factory.mktemp("tiny_laya"))
