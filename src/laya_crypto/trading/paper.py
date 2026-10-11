"""Paper-trading bot: the model trades a simulated account on live public market data.

Nothing here talks to an exchange's order endpoints or uses an API key. It exists to answer the
question a backtest cannot: does the model keep its edge on data it has never seen, in real time?

Execution model (identical to `backtesting.engine`, so paper and backtest results are comparable):
a decision at a candle close fills at the next candle's open, adjusted by half spread + slippage
against the trader, paying the taker fee; stop and target come from the labeling barrier widths
(ATR known at the decision); inside a candle a gap through a level exits at the open, and a
candle touching both stop and target counts as the stop; the vertical barrier exits at the close
of the last holding candle; funding settled while a position is open is charged.

State (account, open positions, pending orders) is saved atomically after every step under
`artifacts/paper/<name>/`, so the bot can be stopped and restarted at any time: it catches up on
the candles it missed for open positions and discards decisions it was too late to act on.
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..config import BotConfig, LabelingConfig, load_bot_config, load_labeling_config, project_path
from ..labeling.triple_barrier import barrier_widths
from ..utils.io import append_jsonl, read_json, read_jsonl, write_json
from ..utils.logging import get_logger
from ..utils.timeframes import tf_ms

log = get_logger("paper")


# ------------------------------------------------------------------------- interfaces
class MarketData(Protocol):
    def closed_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...

    def funding(self, symbol: str) -> pd.DataFrame: ...


class Decider(Protocol):
    def decide(self, symbol: str) -> tuple[dict, pd.Series | None]: ...


class LiveMarketData:
    """Public Binance USD-M market data (read-only REST)."""

    def __init__(self):
        from ..config import load_data_config

        self.cfg = load_data_config()

    def closed_klines(self, symbol, timeframe, limit):
        from ..ingestion.live import fetch_recent_klines

        return fetch_recent_klines(self.cfg, symbol, timeframe, limit)

    def funding(self, symbol):
        from ..ingestion.live import fetch_funding_history

        return fetch_funding_history(self.cfg, symbol, limit=50)


class PredictorDecider:
    """Model decisions through the same inference path as `predict` (LAYA or a baseline)."""

    def __init__(self, bcfg: BotConfig):
        from ..inference.predictor import Predictor

        scorer = None if bcfg.model == "laya" else BaselineScorer(bcfg.model, bcfg.model_dataset)
        self.predictor = Predictor(scorer=scorer)
        self.tf = bcfg.timeframe

    def decide(self, symbol):
        return self.predictor.predict_with_row(symbol, self.tf, offline=False)


class BaselineScorer:
    """A trained baseline (e.g. LightGBM) with its calibration, scoring the live feature row."""

    def __init__(self, name: str, dataset: str | None):
        from ..training.baselines import load_baseline

        if not dataset:
            raise ValueError("bot.model_dataset must name the dataset the baseline was trained on")
        path = project_path("artifacts", "baselines", dataset, f"{name}.pkl")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `baselines --dataset {dataset}` first")
        self.model = load_baseline(path)
        self.version = {"model": name, "dataset": dataset, "artifact": str(path)}
        self.gmm = None
        from ..config import load_data_config

        gmm_path = load_data_config().root_path / "datasets" / dataset / "regime_gmm.pkl"
        if gmm_path.exists():
            import pickle

            with open(gmm_path, "rb") as fh:
                self.gmm = pickle.load(fh)
        elif "regime_gmm" in (getattr(self.model, "features", None) or []):
            log.warning("%s has no regime_gmm.pkl (built by an older version): regime_gmm is left missing "
                        "at inference; rebuild the dataset to fix", dataset)

    def score(self, row: pd.Series, state: str) -> np.ndarray:
        feats = getattr(self.model, "features", None) or list(row.index)
        row = row.copy()
        if "regime_gmm" in feats:
            row["regime_gmm"] = (float(self.gmm.predict(pd.DataFrame([row])).iloc[0]) if self.gmm is not None
                                 else np.nan)
        frame = pd.DataFrame([row.reindex(feats)])  # a feature missing live stays NaN, never invented
        p = self.model.predict_proba(frame.astype(float))
        cal = getattr(self.model, "calibrator", None)
        return (cal.transform(p) if cal is not None else p)[0]


# ------------------------------------------------------------------------------ state
@dataclass
class PaperPosition:
    symbol: str
    side: int
    qty: float
    entry_px: float
    entry_bar: int  # open time of the entry candle
    tp_px: float
    sl_px: float
    max_exit_bar: int  # open time of the last holding candle
    fees: float
    confidence: float
    regime: str = ""
    funding: float = 0.0
    checked_until: int = 0  # open time of the last candle processed for exits
    funding_until: int = 0  # funding settled up to this time


@dataclass
class PaperState:
    cash: float
    peak_equity: float
    day: str
    day_start_equity: float
    started_utc: str
    positions: dict[str, dict] = field(default_factory=dict)
    pending: list[dict] = field(default_factory=list)
    last_decision_bar: int = 0  # close time of the last candle decided on
    halted_reason: str | None = None


class PaperBot:
    def __init__(self, name: str = "default", bcfg: BotConfig | None = None, lcfg: LabelingConfig | None = None,
                 market: MarketData | None = None, decider: Decider | None = None):
        self.name = name
        self.cfg = bcfg or load_bot_config()
        self.lcfg = lcfg or load_labeling_config()
        self.dir = project_path("artifacts", "paper", name)
        self.step_ms = tf_ms(self.cfg.timeframe)
        self.H = self.lcfg.max_hold_bars
        c = self.lcfg.costs
        self.fee = c.taker_fee  # market orders both ways
        self.adverse = (c.half_spread_bps + c.slippage_bps) / 10_000
        self.market = market or LiveMarketData()
        self._decider = decider
        self.state = self._load()

    # ------------------------------------------------------------------ persistence
    @property
    def decider(self) -> Decider:
        if self._decider is None:
            self._decider = PredictorDecider(self.cfg)
        return self._decider

    @property
    def stop_file(self) -> Path:
        return self.dir / "STOP"

    def _load(self) -> PaperState:
        raw = read_json(self.dir / "state.json")
        if raw:
            return PaperState(**raw)
        now = dt.datetime.now(dt.UTC)
        return PaperState(cash=self.cfg.initial_equity, peak_equity=self.cfg.initial_equity,
                          day=now.date().isoformat(), day_start_equity=self.cfg.initial_equity,
                          started_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def save(self) -> None:
        write_json(self.dir / "state.json", asdict(self.state))
        write_json(self.dir / "config.json", {"bot": self.cfg.model_dump(), "costs": self.lcfg.costs.model_dump(),
                                              "max_hold_bars": self.H})

    def event(self, kind: str, **kw) -> None:
        append_jsonl(self.dir / "journal.jsonl", {"ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                  "event": kind, **kw})

    # ------------------------------------------------------------------ one step
    def step(self, now_ms: int | None = None) -> dict:
        """Process everything that is known at `now_ms` (default: the wall clock)."""
        now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        last_close = (now_ms // self.step_ms) * self.step_ms  # close time of the latest closed candle
        candles = {}
        symbols = set(self.state.positions) | {o["symbol"] for o in self.state.pending}
        for sym in symbols:
            try:
                k = self.market.closed_klines(sym, self.cfg.timeframe, min(1500, self.H + 300))
                candles[sym] = k[k["open_time"] + self.step_ms <= last_close].set_index("open_time")
            except Exception as e:  # noqa: BLE001 - one symbol's data problem must not stop the bot
                self.event("data_error", symbol=sym, error=repr(e))
        self._fill_pending(candles, last_close)
        self._manage_positions(candles)
        self._settle_funding(last_close)
        equity = self._mark_to_market(candles)
        self._risk_state(equity, last_close)
        decisions = []
        if last_close > self.state.last_decision_bar:
            decisions = self._decide(last_close, equity)
            self.state.last_decision_bar = last_close
        append_jsonl(self.dir / "equity.jsonl", {"time": last_close, "equity": round(equity, 6),
                                                 "cash": round(self.state.cash, 6),
                                                 "open_positions": len(self.state.positions)})
        self.save()
        status = {"name": self.name, "time_utc": str(pd.to_datetime(last_close, unit="ms", utc=True)),
                  "equity": equity, "cash": self.state.cash, "positions": self.state.positions,
                  "pending": self.state.pending, "halted_reason": self.state.halted_reason,
                  "decisions": decisions, "model": self.cfg.model}
        write_json(self.dir / "status.json", status)
        return status

    def _fill_pending(self, candles: dict, last_close: int) -> None:
        keep = []
        for od in self.state.pending:
            entry_bar = od["decision_close"]  # the candle that opens at the decision time
            k = candles.get(od["symbol"])
            if k is None or entry_bar not in k.index:
                if entry_bar + self.step_ms <= last_close - self.step_ms:
                    self.event("order_expired", symbol=od["symbol"], reason="entry candle unavailable")
                else:
                    keep.append(od)  # entry candle not closed yet
                continue
            o = float(k.at[entry_bar, "open"])
            side = od["side"]
            px = o * (1 + side * self.adverse)
            risk_cash = self.state.cash * self.cfg.risk_per_trade
            per_unit_loss = od["sl"] + 2 * self.adverse + 2 * self.fee
            notional = min(risk_cash / per_unit_loss,
                           self.state.cash * self.cfg.max_position_notional_frac * self.cfg.max_leverage)
            margin_used = sum(p["qty"] * p["entry_px"] / self.cfg.max_leverage for p in self.state.positions.values())
            if margin_used + notional / self.cfg.max_leverage > self.state.cash or notional <= 0:
                self.event("order_rejected", symbol=od["symbol"], reason="insufficient margin")
                continue
            qty = notional / px
            fee = qty * px * self.fee
            self.state.cash -= fee
            pos = PaperPosition(od["symbol"], side, qty, px, entry_bar, px * (1 + side * od["tp"]),
                                px * (1 - side * od["sl"]), entry_bar + (self.H - 1) * self.step_ms, fee,
                                od["confidence"], od.get("regime", ""), checked_until=entry_bar - self.step_ms)
            self.state.positions[od["symbol"]] = asdict(pos)
            self.event("fill", symbol=pos.symbol, side="LONG" if side > 0 else "SHORT", price=px, qty=qty,
                       notional=qty * px, fee=fee, tp=pos.tp_px, sl=pos.sl_px)
        self.state.pending = keep

    def _manage_positions(self, candles: dict) -> None:
        for sym in list(self.state.positions):
            p = PaperPosition(**self.state.positions[sym])
            k = candles.get(sym)
            if k is None:
                continue
            done = False
            for t, bar in k[k.index > p.checked_until].iterrows():
                exit_px, reason = self._exit_in_bar(p, bar, int(t))
                p.checked_until = int(t)
                if reason:
                    self._close(p, exit_px, int(t) + (self.step_ms if reason == "timeout" else 0), reason)
                    done = True
                    break
            if not done:
                self.state.positions[sym] = asdict(p)

    def _exit_in_bar(self, p: PaperPosition, bar, t: int) -> tuple[float | None, str | None]:
        o, h, lo, c = (float(bar[x]) for x in ("open", "high", "low", "close"))
        s = p.side
        if (s > 0 and o <= p.sl_px) or (s < 0 and o >= p.sl_px):
            return o, "stop_gap"
        if (s > 0 and lo <= p.sl_px) or (s < 0 and h >= p.sl_px):
            return p.sl_px, "stop"  # also when the target was touched in the same candle
        if (s > 0 and o >= p.tp_px) or (s < 0 and o <= p.tp_px):
            return o, "take_profit_gap"
        if (s > 0 and h >= p.tp_px) or (s < 0 and lo <= p.tp_px):
            return p.tp_px, "take_profit"
        if t >= p.max_exit_bar:
            return c, "timeout"
        return None, None

    def _close(self, p: PaperPosition, px: float, t_exit: int, reason: str) -> None:
        exit_px = px * (1 - p.side * self.adverse)
        fee = p.qty * exit_px * self.fee
        gross = p.side * p.qty * (exit_px - p.entry_px)
        self.state.cash += gross - fee
        net = gross - p.fees - fee - p.funding
        del self.state.positions[p.symbol]
        self.event("exit", symbol=p.symbol, side="LONG" if p.side > 0 else "SHORT", reason=reason,
                   entry_px=p.entry_px, exit_px=exit_px, qty=p.qty, notional=p.qty * p.entry_px, gross_pnl=gross,
                   fees=p.fees + fee, funding=p.funding, net_pnl=net, entry_bar=p.entry_bar, exit_time=t_exit,
                   hold_bars=int((t_exit - p.entry_bar) // self.step_ms), confidence=p.confidence, regime=p.regime)

    def _settle_funding(self, last_close: int) -> None:
        for sym, raw in list(self.state.positions.items()):
            try:
                f = self.market.funding(sym)
            except Exception as e:  # noqa: BLE001
                self.event("data_error", symbol=sym, error=f"funding: {e!r}")
                continue
            paid_until = raw.get("funding_until") or raw["entry_bar"]
            due = f[(f["funding_time"] > paid_until) & (f["funding_time"] <= last_close)]
            if len(due):
                pay = raw["side"] * raw["qty"] * raw["entry_px"] * float(due["funding_rate"].sum())
                raw["funding"] += pay
                self.state.cash -= pay
                self.event("funding", symbol=sym, amount=pay, settlements=len(due))
            raw["funding_until"] = last_close

    def _mark_to_market(self, candles: dict) -> float:
        unreal = 0.0
        for sym, raw in self.state.positions.items():
            k = candles.get(sym)
            px = float(k["close"].iloc[-1]) if k is not None and len(k) else raw["entry_px"]
            unreal += raw["side"] * raw["qty"] * (px - raw["entry_px"])
        return self.state.cash + unreal

    def _risk_state(self, equity: float, last_close: int) -> None:
        st = self.state
        day = pd.to_datetime(last_close, unit="ms", utc=True).date().isoformat()
        if day != st.day:
            st.day, st.day_start_equity = day, equity
            if st.halted_reason == "daily_loss":
                st.halted_reason = None
                self.event("resume", reason="new UTC day")
        st.peak_equity = max(st.peak_equity, equity)
        if st.halted_reason is None:
            if equity <= st.day_start_equity * (1 - self.cfg.max_daily_loss):
                st.halted_reason = "daily_loss"
            elif equity <= st.peak_equity * (1 - self.cfg.max_drawdown):
                st.halted_reason = "max_drawdown"
            if st.halted_reason:
                self.event("halt", reason=st.halted_reason, equity=equity)

    def _decide(self, last_close: int, equity: float) -> list[dict]:
        out = []
        for sym in self.cfg.symbols:
            res, row = self.decider.decide(sym)
            rec = {"symbol": sym, "decision": res.get("decision"), "probabilities": res.get("probabilities"),
                   "error": res.get("error"), "data_timestamp_utc": res.get("data_timestamp_utc")}
            veto = self._veto(sym, res, row, last_close)
            rec["veto"] = veto
            if veto is None:
                atr = float(row["atr_pct"])
                tp, sl = barrier_widths(np.array([atr]), self.lcfg)
                order = {"symbol": sym, "side": 1 if res["decision"] == "LONG" else -1, "tp": float(tp[0]),
                         "sl": float(sl[0]), "decision_close": last_close,
                         "confidence": max(res["probabilities"].values()),
                         "regime": str(row.get("regime_trend", ""))}
                self.state.pending.append(order)
                rec["order"] = order
            self.event("decision", **rec)
            out.append(rec)
        return out

    def _veto(self, sym: str, res: dict, row, last_close: int) -> str | None:
        if res.get("error"):
            return "no_prediction"
        if res.get("decision") not in ("LONG", "SHORT"):
            return "wait"
        if res.get("stale"):
            return "stale_data"
        expected_close = res.get("data_timestamp_utc")
        if expected_close and pd.Timestamp(expected_close).value // 1_000_000 != last_close:
            return "prediction_not_for_latest_candle"
        if self.state.halted_reason:
            return f"halted:{self.state.halted_reason}"
        if max(res["probabilities"].values()) < self.cfg.min_confidence:
            return "low_confidence"
        if self.cfg.veto_abnormal and row is not None and row.get("regime_abnormal") == 1:
            return "abnormal_regime"
        if sym in self.state.positions or any(o["symbol"] == sym for o in self.state.pending):
            return "already_in_symbol"
        if len(self.state.positions) + len(self.state.pending) >= self.cfg.max_positions:
            return "max_positions"
        if row is None or not np.isfinite(float(row.get("atr_pct", np.nan))):
            return "no_atr"
        return None

    # ------------------------------------------------------------------ loop
    def run(self, max_steps: int | None = None) -> None:
        self.stop_file.unlink(missing_ok=True)
        self.event("start", model=self.cfg.model, symbols=self.cfg.symbols)
        n = 0
        while max_steps is None or n < max_steps:
            now = time.time() * 1000
            next_close = (now // self.step_ms + 1) * self.step_ms
            wake = next_close + self.cfg.bar_delay_seconds * 1000
            while time.time() * 1000 < wake:
                if self.stop_file.exists():
                    break
                time.sleep(min(5.0, max(0.0, (wake - time.time() * 1000) / 1000)))
            if self.stop_file.exists():
                break
            try:
                st = self.step()
                log.info("%s equity %.2f positions %d pending %d %s", st["time_utc"], st["equity"],
                         len(st["positions"]), len(st["pending"]), st["halted_reason"] or "")
            except Exception as e:  # noqa: BLE001 - keep running; the error is journaled
                log.exception("step failed")
                self.event("step_error", error=repr(e))
            n += 1
        self.save()
        self.event("stop", open_positions=list(self.state.positions))
        log.info("paper bot stopped; state saved (open positions are kept and managed on restart)")


def report(name: str = "default") -> dict:
    """Performance of a paper-trading account so far, from its journal and equity log."""
    from ..backtesting.engine import performance

    d = project_path("artifacts", "paper", name)
    state = read_json(d / "state.json")
    if not state:
        raise FileNotFoundError(f"no paper account named {name!r}")
    cfg = read_json(d / "config.json") or {}
    initial = (cfg.get("bot") or {}).get("initial_equity", 2000.0)
    exits = [e for e in read_jsonl(d / "journal.jsonl") if e["event"] == "exit"]
    eq = pd.DataFrame(read_jsonl(d / "equity.jsonl"))
    if not eq.empty:
        eq = eq.drop_duplicates("time", keep="last").sort_values("time")
        eq["drawdown"] = 1 - eq["equity"] / eq["equity"].cummax().clip(lower=initial)
    trades = pd.DataFrame(exits)
    res = {"trades": trades, "equity": eq if not eq.empty else pd.DataFrame({"time": [], "equity": [], "drawdown": []}),
           "vetoes": {}, "exposure": float((eq["open_positions"] > 0).mean()) if not eq.empty else 0.0}
    perf = performance(res, initial) if not eq.empty else {"trades": 0}
    started = pd.Timestamp(state["started_utc"])
    perf.update(name=name, started_utc=state["started_utc"],
                days_running=round((pd.Timestamp.now(tz="UTC") - started).total_seconds() / 86400, 2),
                open_positions=list(state["positions"]), halted_reason=state.get("halted_reason"),
                note="paper trading on live data: simulated fills, no orders were sent")
    write_json(d / "report.json", perf)
    return perf


LEARNED_BASELINES = ("lightgbm", "xgboost", "logistic_regression")


def _dataset_dir(dataset: str) -> Path:
    from ..config import load_data_config

    return load_data_config().root_path / "datasets" / dataset


def choose_model(dataset: str, write_config: bool = True) -> dict:
    """Pick the decider for the bot on VALIDATION log loss (never test) and write it to bot.yaml.

    Candidates: the learned baselines of `dataset`, plus LAYA (`models/best`) if it was scored on
    the same dataset. The test split is reported for information only.
    """
    import re

    from ..config import CONFIG_DIR
    from ..evaluation.scoring import comparison_table

    tab = comparison_table(dataset)
    if tab.empty:
        raise FileNotFoundError(f"no scored models for dataset {dataset!r}: run `baselines --dataset {dataset}`")
    val = tab[tab["split"] == "val"]
    best_laya = read_json(project_path("models", "best", "selection.json"))
    allowed = set(LEARNED_BASELINES) | ({f"laya-{best_laya['run_id']}"} if best_laya else set())
    val = val[val["model"].isin(allowed)].sort_values("log_loss")
    if val.empty:
        raise FileNotFoundError("no learned model with validation scores")
    pick = val.iloc[0]
    model = "laya" if str(pick["model"]).startswith("laya-") else str(pick["model"])
    test = tab[(tab["split"] == "test") & (tab["model"] == pick["model"])]
    out = {"model": model, "model_dataset": dataset, "chosen_by": "lowest validation log loss",
           "val": {k: float(pick[k]) if pick[k] is not None else None
                   for k in ("log_loss", "accuracy", "expectancy")},
           "test_for_information": test[["log_loss", "accuracy", "expectancy"]].iloc[0].to_dict() if len(test) else None,
           "candidates": val[["model", "log_loss", "accuracy", "expectancy"]].to_dict("records")}
    if write_config:
        path = CONFIG_DIR / "bot.yaml"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        manifest = read_json(_dataset_dir(dataset) / "manifest.json") or {}
        symbols = manifest.get("symbols") or []
        pairs = [("model", model), ("model_dataset", dataset)]
        if symbols:  # trade the symbols the model was trained on
            pairs.append(("symbols", "[" + ", ".join(symbols) + "]"))
            out["symbols"] = symbols
        for key, value in pairs:
            line = f"{key}: {value}"
            if re.search(rf"^{key}:.*$", text, flags=re.M):
                text = re.sub(rf"^{key}:[^\n#]*", line + " ", text, count=1, flags=re.M)
            else:
                text += f"\n{line}\n"
        path.write_text(text, encoding="utf-8")
        out["written_to"] = str(path)
    return out
