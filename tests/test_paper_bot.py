"""Paper-trading bot on a fake market: fills, exits, costs, risk vetoes, persistence, stop."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from laya_crypto.config import BotConfig, CostsConfig, LabelingConfig
from laya_crypto.trading import paper
from laya_crypto.trading.paper import PaperBot

STEP = 300_000
T0 = 1_760_000_000_000 // STEP * STEP
FLAT = (100, 100.2, 99.8, 100)


class FakeMarket:
    def __init__(self):
        self.bars: dict[str, list] = {}

    def add(self, sym, t, ohlc):
        self.bars.setdefault(sym, []).append((t, *ohlc))

    def closed_klines(self, symbol, timeframe, limit):
        return pd.DataFrame(self.bars.get(symbol, []), columns=["open_time", "open", "high", "low", "close"])

    def funding(self, symbol):
        return pd.DataFrame({"funding_time": pd.Series([], dtype="int64"), "funding_rate": pd.Series([], dtype=float)})


class FakeDecider:
    def __init__(self):
        self.plan: dict[str, tuple[str, float]] = {}
        self.close_ms = 0

    def decide(self, symbol):
        decision, conf = self.plan.get(symbol, ("WAIT", 0.9))
        probs = {"LONG": 0.05, "SHORT": 0.05, "WAIT": 0.05}
        probs[decision] = conf
        res = {"symbol": symbol, "decision": decision, "probabilities": probs, "stale": False,
               "data_timestamp_utc": str(pd.to_datetime(self.close_ms, unit="ms", utc=True))}
        return res, pd.Series({"atr_pct": 0.005, "regime_trend": "SIDEWAYS", "regime_abnormal": 0})


@pytest.fixture
def bot_env(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "project_path", lambda *p: tmp_path.joinpath(*p))
    bcfg = BotConfig(symbols=["X"], min_confidence=0.5, risk_per_trade=0.01, max_leverage=3, max_positions=2,
                     max_position_notional_frac=1.0, initial_equity=2000, max_daily_loss=0.5, max_drawdown=0.9)
    lcfg = LabelingConfig(method="triple_barrier_pct", tp_pct=0.02, sl_pct=0.01, max_hold_bars=3, min_barrier_pct=0,
                          costs=CostsConfig(taker_fee=0.0005, half_spread_bps=1, slippage_bps=2))
    market, decider = FakeMarket(), FakeDecider()

    def make():
        return PaperBot("t", bcfg, lcfg, market, decider)
    return make, market, decider, tmp_path


def run_step(bot, decider, close_ms):
    decider.close_ms = close_ms
    return bot.step(now_ms=close_ms + 1000)


def test_long_take_profit_with_exact_costs(bot_env):
    make, market, decider, _ = bot_env
    bot = make()
    decider.plan["X"] = ("LONG", 0.9)
    market.add("X", T0 - STEP, FLAT)
    st = run_step(bot, decider, T0)  # decision at the close of the candle ending at T0
    assert len(st["pending"]) == 1 and st["decisions"][0]["veto"] is None
    decider.plan["X"] = ("WAIT", 0.9)
    market.add("X", T0, (100, 103, 99.6, 102))  # entry candle opens at 100 and reaches +2%
    st = run_step(bot, decider, T0 + STEP)
    exits = [e for e in paper.read_jsonl(bot.dir / "journal.jsonl") if e["event"] == "exit"]
    assert len(exits) == 1 and exits[0]["reason"] == "take_profit"
    adv = 3 / 10_000
    entry = 100 * (1 + adv)
    exit_px = entry * 1.02 * (1 - adv)
    qty = exits[0]["qty"]
    expected = qty * (exit_px - entry) - qty * entry * 0.0005 - qty * exit_px * 0.0005
    assert np.isclose(exits[0]["net_pnl"], expected) and np.isclose(st["cash"], 2000 + expected)
    assert st["positions"] == {}


def test_stop_first_when_both_touched_and_timeout(bot_env):
    make, market, decider, _ = bot_env
    bot = make()
    decider.plan["X"] = ("SHORT", 0.9)
    market.add("X", T0 - STEP, FLAT)
    run_step(bot, decider, T0)
    decider.plan["X"] = ("WAIT", 0.9)
    market.add("X", T0, (100, 101.5, 97.5, 100))  # touches the short's stop (+1%) and target (-2%)
    run_step(bot, decider, T0 + STEP)
    ex = [e for e in paper.read_jsonl(bot.dir / "journal.jsonl") if e["event"] == "exit"][-1]
    assert ex["reason"] == "stop" and ex["net_pnl"] < 0
    decider.plan["X"] = ("LONG", 0.9)
    run_step(bot, decider, T0 + 2 * STEP)
    decider.plan["X"] = ("WAIT", 0.9)
    for i in range(2, 6):
        market.add("X", T0 + i * STEP, FLAT)
        run_step(bot, decider, T0 + (i + 1) * STEP)
    ex = [e for e in paper.read_jsonl(bot.dir / "journal.jsonl") if e["event"] == "exit"][-1]
    assert ex["reason"] == "timeout" and ex["hold_bars"] == 3


def test_vetoes(bot_env):
    make, market, decider, _ = bot_env
    bot = make()
    decider.plan["X"] = ("LONG", 0.4)
    market.add("X", T0 - STEP, FLAT)
    st = run_step(bot, decider, T0)
    assert st["decisions"][0]["veto"] == "low_confidence"
    decider.plan["X"] = ("LONG", 0.9)
    decider.close_ms = T0 - STEP  # model answered for an older candle
    st = bot.step(now_ms=T0 + STEP + 1000)
    assert st["decisions"][0]["veto"] == "prediction_not_for_latest_candle"
    bot.state.halted_reason = "max_drawdown"
    st = run_step(bot, decider, T0 + 2 * STEP)
    assert st["decisions"][0]["veto"] == "halted:max_drawdown" and not st["pending"]


def test_daily_loss_halts_new_trades(bot_env):
    make, market, decider, _ = bot_env
    bot = make()
    bot.cfg = bot.cfg.model_copy(update={"max_daily_loss": 0.001})
    bot.state.day = pd.to_datetime(T0, unit="ms", utc=True).date().isoformat()
    bot.state.cash = 1990.0  # -0.5% today
    decider.plan["X"] = ("LONG", 0.9)
    st = run_step(bot, decider, T0)
    assert st["halted_reason"] == "daily_loss" and st["decisions"][0]["veto"] == "halted:daily_loss"


def test_state_survives_restart(bot_env):
    make, market, decider, _ = bot_env
    bot = make()
    decider.plan["X"] = ("LONG", 0.9)
    market.add("X", T0 - STEP, FLAT)
    run_step(bot, decider, T0)
    decider.plan["X"] = ("WAIT", 0.9)
    market.add("X", T0, FLAT)
    run_step(bot, decider, T0 + STEP)
    assert "X" in bot.state.positions
    restarted = make()  # e.g. the PC rebooted
    assert "X" in restarted.state.positions
    market.add("X", T0 + STEP, (100, 100.1, 98.5, 99))  # stop hit while the bot was off
    market.add("X", T0 + 2 * STEP, FLAT)
    run_step(restarted, decider, T0 + 3 * STEP)  # catches up on both missed candles
    ex = [e for e in paper.read_jsonl(restarted.dir / "journal.jsonl") if e["event"] == "exit"]
    assert ex and ex[-1]["reason"] == "stop" and not restarted.state.positions
    rep = paper.report("t")
    assert rep["trades"] == 1 and "return_pct" in rep


def test_stop_file_ends_the_loop(bot_env):
    make, market, decider, _ = bot_env
    bot = make()
    threading.Timer(0.5, lambda: bot.stop_file.write_text("stop")).start()
    t = time.time()
    bot.run()
    assert time.time() - t < 10
    assert paper.read_jsonl(bot.dir / "journal.jsonl")[-1]["event"] == "stop"


def test_dashboard_paper_endpoints(bot_env, monkeypatch):
    from fastapi.testclient import TestClient

    from laya_crypto.dashboard import app as appmod

    make, market, decider, tmp = bot_env
    monkeypatch.setattr(appmod, "project_path", lambda *p: tmp.joinpath(*p))
    bot = make()
    decider.plan["X"] = ("LONG", 0.9)
    market.add("X", T0 - STEP, FLAT)
    run_step(bot, decider, T0)
    c = TestClient(appmod.app)
    assert c.get("/api/paper").json() == ["t"]
    d = c.get("/api/paper/t").json()
    assert d["status"]["pending"] and d["events"][0]["event"] == "decision"
    assert c.get("/api/paper/../../x").status_code == 404
    assert c.post("/api/paper/t/stop").status_code == 200 and bot.stop_file.exists()


def test_choose_model_uses_validation_and_writes_config(tmp_path, monkeypatch):
    import laya_crypto.config as config
    from laya_crypto.evaluation import scoring
    from laya_crypto.utils.io import write_json

    monkeypatch.setattr(scoring, "project_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(paper, "project_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(paper, "_dataset_dir", lambda name: tmp_path / "data" / "datasets" / name)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "configs")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "bot.yaml").write_text("model: laya   # comment kept\nmodel_dataset: null\nsymbols: [X]\n")
    write_json(tmp_path / "data" / "datasets" / "d" / "manifest.json", {"symbols": ["AUSDT", "BUSDT"]})

    def rep(model, val_ll, test_ll):
        cls = {"accuracy": 0.4, "balanced_accuracy": 0.4, "macro_f1": 0.4, "brier": 0.6, "ece": 0.01, "n": 10}
        sig = {"trades": 1, "expectancy": 0.001, "win_rate": 0.5, "profit_factor": 1.0, "net_return_sum": 0.0,
               "per_decision_mean_ci": {"ci_low": 0, "ci_high": 0}}
        return {"model": model, "splits": {"val": {"classification": {**cls, "log_loss": val_ll}, "signals": sig},
                                           "test": {"classification": {**cls, "log_loss": test_ll}, "signals": sig}}}
    write_json(tmp_path / "reports" / "d" / "lightgbm.json", rep("lightgbm", 1.09, 1.20))
    write_json(tmp_path / "reports" / "d" / "xgboost.json", rep("xgboost", 1.10, 1.00))  # better on TEST only
    write_json(tmp_path / "reports" / "d" / "always_wait.json", rep("always_wait", 0.5, 0.5))  # not a candidate
    out = paper.choose_model("d")
    assert out["model"] == "lightgbm"  # chosen on validation, test is ignored for the choice
    text = (tmp_path / "configs" / "bot.yaml").read_text()
    assert "model: lightgbm" in text and "model_dataset: d" in text and "symbols: [AUSDT, BUSDT]" in text
    assert "# comment kept" in text
