"""Backtest fees/slippage/ordering, metrics, calibration and the point-in-time universe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from laya_crypto.backtesting.engine import Backtester, SymbolBars, performance
from laya_crypto.config import BacktestConfig, CostsConfig, DataConfig, LabelingConfig, UniverseConfig
from laya_crypto.evaluation import metrics
from laya_crypto.ingestion.universe import build_universe, candidate_symbols, membership_mask
from laya_crypto.training.baselines import TemperatureCalibrator

STEP = 300_000


def bars(rows, atr=0.01, funding=None):
    a = np.array(rows, dtype=float)
    b = SymbolBars(0, a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.full(len(a), atr), step=STEP)
    if funding is not None:
        b.funding_t, b.funding_r = np.array(funding[0], dtype="int64"), np.array(funding[1], dtype=float)
    return b


def lcfg(**kw):
    c = dict(method="triple_barrier_pct", tp_pct=0.02, sl_pct=0.01, max_hold_bars=3, min_barrier_pct=0.0,
             costs=CostsConfig(taker_fee=0.0005, half_spread_bps=1, slippage_bps=2, include_funding=True))
    c.update(kw)
    return LabelingConfig(**c)


BCFG = BacktestConfig(initial_equity_usdt=2000, risk_per_trade=0.01, max_leverage=3, max_positions=5,
                      max_position_notional_frac=1.0, max_portfolio_drawdown=0.9)
FLAT = [100, 100.2, 99.8, 100]


def signal(t_bar, pred="LONG", sym="X"):
    return pd.DataFrame({"decision_time": [t_bar * STEP + STEP], "symbol": [sym], "pred": [pred], "confidence": [0.9],
                         "regime_trend": ["SIDEWAYS"], "regime_abnormal": [0]})


def test_take_profit_trade_costs_exact():
    b = bars([FLAT, [100, 100.5, 99.5, 100], [100, 103, 99.6, 102], FLAT, FLAT, FLAT])
    res = Backtester({"X": b}, BCFG, lcfg()).run(signal(0))
    t = res["trades"].iloc[0]
    adverse = 3 / 10_000
    entry = 100 * (1 + adverse)
    assert t["reason"] == "take_profit" and np.isclose(t["entry_px"], entry)
    exit_px = entry * 1.02 * (1 - adverse)
    assert np.isclose(t["exit_px"], exit_px)
    qty = t["qty"]
    expected = qty * (exit_px - entry) - qty * entry * 0.0005 - qty * exit_px * 0.0005
    assert np.isclose(t["net_pnl"], expected)
    perf = performance(res, 2000)
    assert np.isclose(perf["final_equity"], 2000 + expected)


def test_both_touched_assumes_stop_and_gap_exits_at_open():
    b = bars([FLAT, [100, 103, 98, 100], FLAT, FLAT, FLAT])
    t = Backtester({"X": b}, BCFG, lcfg()).run(signal(0))["trades"].iloc[0]
    assert t["reason"] == "stop" and t["net_pnl"] < 0
    gap = bars([FLAT, [100, 100.2, 99.9, 100], [95, 95.5, 94, 95], FLAT, FLAT])
    t2 = Backtester({"X": gap}, BCFG, lcfg()).run(signal(0))["trades"].iloc[0]
    assert t2["reason"] == "stop_gap" and np.isclose(t2["exit_px"], 95 * (1 - 3 / 10_000))


def test_funding_charged_to_longs_and_risk_vetoes():
    b = bars([FLAT] * 8, funding=([STEP * 2 + 1], [0.001]))
    res = Backtester({"X": b}, BCFG, lcfg()).run(signal(0))
    t = res["trades"].iloc[0]
    assert t["reason"] == "timeout" and t["funding"] > 0
    sig = pd.concat([signal(0), signal(0, "SHORT")])
    res2 = Backtester({"X": bars([FLAT] * 8)}, BCFG, lcfg()).run(sig)
    assert res2["vetoes"].get("already_in_symbol") == 1 and len(res2["trades"]) == 1
    low = signal(0).assign(confidence=0.3)
    res3 = Backtester({"X": bars([FLAT] * 8)}, BCFG.model_copy(update={"min_confidence": 0.5}), lcfg()).run(low)
    assert res3["vetoes"] == {"low_confidence": 1} and res3["trades"].empty


def test_classification_metrics_and_ece():
    y = np.array(["LONG", "SHORT", "WAIT"] * 100)
    perfect = np.eye(3)[np.tile([0, 1, 2], 100)]
    rep = metrics.classification_report(y, perfect)
    assert rep["accuracy"] == 1 and rep["ece"] < 1e-6 and rep["macro_f1"] == 1
    uniform = np.full((300, 3), 1 / 3)
    rep2 = metrics.classification_report(y, uniform)
    assert np.isclose(rep2["log_loss"], np.log(3)) and np.isclose(rep2["brier"], 2 / 3)


def test_block_bootstrap_brackets_mean():
    x = np.random.default_rng(0).normal(0.001, 0.01, 5000)
    ci = metrics.block_bootstrap_mean(x, 100, 300)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]


def test_temperature_calibrator_softens_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 3000)
    logits = rng.normal(0, 1, (3000, 3))
    logits[np.arange(3000), y] += 1.0
    p = np.exp(4 * logits)
    p /= p.sum(1, keepdims=True)  # sharpened 4x -> overconfident
    cal = TemperatureCalibrator().fit(p, y)
    assert 3 < cal.T < 5
    labels = np.array(["LONG", "SHORT", "WAIT"])[y]
    assert metrics.ece(cal.transform(p), labels)[0] < metrics.ece(p, labels)[0]


def test_universe_is_point_in_time():
    days = pd.date_range("2024-01-01", "2024-06-30", freq="D", tz="UTC")
    panel = pd.DataFrame(index=days)
    panel["OLDUSDT"] = 50e6
    panel["NEWUSDT"] = np.where(days >= pd.Timestamp("2024-04-10", tz="UTC"), 500e6, np.nan)
    panel["DEADUSDT"] = np.where(days < pd.Timestamp("2024-05-15", tz="UTC"), 80e6, np.nan)
    cfg = DataConfig(universe=UniverseConfig(top_n=2, min_history_days=30, rank_window_days=30,
                                             min_avg_daily_quote_volume=1e6))
    uni = build_universe(cfg, panel, pd.Timestamp("2024-03-01").date(), pd.Timestamp("2024-06-30").date())
    month = lambda m: set(uni.loc[uni["month"] == pd.Timestamp(m, tz="UTC"), "symbol"])  # noqa: E731
    assert month("2024-03-01") == {"OLDUSDT", "DEADUSDT"}
    assert "NEWUSDT" not in month("2024-05-01")  # only 21 days of history before May
    assert "NEWUSDT" in month("2024-06-01")
    assert "DEADUSDT" in month("2024-05-01") and "DEADUSDT" not in month("2024-06-01")  # delisted, kept in the past
    mask = membership_mask(uni, "DEADUSDT", pd.Series([pd.Timestamp("2024-05-03", tz="UTC").value // 10**6,
                                                       pd.Timestamp("2024-06-03", tz="UTC").value // 10**6]))
    assert mask.tolist() == [True, False]
    assert candidate_symbols(["BTCUSDT", "ETHUSDT_240329", "BTCUSDC", "USDCUSDT", "1000PEPEUSDT"], cfg) == \
        ["1000PEPEUSDT", "BTCUSDT"]


@pytest.mark.parametrize("vram,tier,micro", [(6.0, "low_vram", 2), (12.0, "medium_vram", 4), (24.0, "high_vram", 8)])
def test_hardware_profile_selection(vram, tier, micro):
    from laya_crypto.config import HardwareConfig
    from laya_crypto.hardware.detect import select_profile

    rep = {"cpu": {"logical_threads": 16}, "torch": {"cuda_available": True, "cuda_matmul_ok": True,
                                                     "devices": [{"name": "NVIDIA GeForce RTX", "total_gb": vram,
                                                                  "capability": "8.6"}]}}
    p = select_profile(rep, HardwareConfig())
    assert p["tier"] == tier and p["micro_batch"] == micro and p["micro_batch"] * p["grad_accum"] == 64
    rep["torch"]["devices"][0]["name"] = "NVIDIA GeForce GTX 1660"
    assert select_profile(rep, HardwareConfig())["amp"] is False
    assert select_profile({"cpu": {"logical_threads": 4}, "torch": {"cuda_available": False}})["device"] == "cpu"


def test_hardware_inspect_runs():
    from laya_crypto.hardware.detect import inspect, to_markdown

    rep = inspect(".")
    assert {"os", "cpu", "memory", "nvidia", "torch", "disk", "profile"} <= set(rep)
    assert "Hardware report" in to_markdown(rep)
