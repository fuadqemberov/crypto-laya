"""Event-driven, causally ordered portfolio backtest.

Per base bar, in this order:

1. **Fills**: orders decided at the previous bar's close fill at this bar's open, adjusted by
   half-spread + slippage against the trader, paying the configured fee.
2. **Exits** from this bar's range, with the order of events inside a bar unknown, resolved
   conservatively: a gap through the stop or liquidation price exits at the open. If the bar
   touches both the stop and the target, the stop is assumed. Liquidation (isolated margin,
   maintenance margin rate) is checked before the stop. The vertical barrier exits at the close of
   the last holding bar.
3. **Funding** settled at its timestamps on open positions (longs pay positive rates).
4. **Decisions** at this bar's close: the model's signal, then the risk engine, which may veto
   (position limits, one position per symbol, drawdown halt, confidence floor, abnormal regime,
   exchange filters, margin). Approved orders fill at the next bar's open (step 1).

The barrier widths are the labeling widths (ATR-based, from data known at the decision bar), so
backtest and labels describe the same trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig, LabelingConfig
from ..labeling.triple_barrier import barrier_widths
from ..utils.timeframes import tf_ms


@dataclass
class Position:
    symbol: str
    side: int
    qty: float
    entry_px: float
    entry_time: int
    entry_idx: int
    tp_px: float
    sl_px: float
    liq_px: float
    margin: float
    max_exit_idx: int
    fees: float
    regime: str = ""
    confidence: float = 0.0
    funding: float = 0.0


@dataclass
class SymbolBars:
    t0: int
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr_pct: np.ndarray
    funding_t: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="int64"))
    funding_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    step: int = 300_000

    def idx(self, t: int) -> int:
        return (t - self.t0) // self.step

    def __len__(self):
        return len(self.open)


def make_bars(frame: pd.DataFrame, funding: pd.DataFrame | None, timeframe: str) -> SymbolBars:
    """`frame` is a full-grid feature frame (data/features/<tf>/<SYMBOL>.parquet)."""
    f = frame.sort_values("open_time")
    fb = SymbolBars(int(f["open_time"].iloc[0]), *(f[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")),
                    f["atr_pct"].to_numpy(dtype=float), step=tf_ms(timeframe))
    if funding is not None and len(funding):
        fs = funding.sort_values("funding_time")
        fb.funding_t, fb.funding_r = fs["funding_time"].to_numpy("int64"), fs["funding_rate"].to_numpy(float)
    return fb


class Backtester:
    def __init__(self, bars: dict[str, SymbolBars], cfg: BacktestConfig, lcfg: LabelingConfig, timeframe: str = "5m",
                 exchange_info: dict | None = None, veto_abnormal: bool = True):
        self.bars, self.cfg, self.lcfg = bars, cfg, lcfg
        self.step = tf_ms(timeframe)
        self.H = lcfg.max_hold_bars
        c = lcfg.costs
        self.entry_fee = c.taker_fee if c.entry_order == "taker" else c.maker_fee
        self.exit_fee = c.taker_fee if c.exit_order == "taker" else c.maker_fee
        self.adverse = (c.half_spread_bps + c.slippage_bps) / 10_000
        self.filters = (exchange_info or {}).get("symbols", {})
        self.veto_abnormal = veto_abnormal

    def _filters_ok(self, symbol: str, qty: float, px: float) -> tuple[float, str | None]:
        f = self.filters.get(symbol)
        if not f:
            return qty, None
        step = f.get("step_size")
        if step and np.isfinite(step) and step > 0:
            qty = np.floor(qty / step) * step
        mn = f.get("min_notional")
        if qty <= 0 or (mn and np.isfinite(mn) and qty * px < mn):
            return 0.0, "below_exchange_min_notional"
        return qty, None

    def run(self, signals: pd.DataFrame) -> dict:
        """`signals`: decision_time, symbol, pred, confidence, regime_trend, regime_abnormal (optional)."""
        cfg = self.cfg
        sig = signals.sort_values(["decision_time", "symbol"])
        by_time = {t: g for t, g in sig.groupby("decision_time")}
        t_start = int(sig["decision_time"].min()) - self.step  # the bar whose close is the first decision
        t_end = int(sig["decision_time"].max()) + (self.H + 1) * self.step
        timeline = np.arange(t_start, t_end + self.step, self.step, dtype="int64")  # bar open times
        cash, peak = cfg.initial_equity_usdt, cfg.initial_equity_usdt
        positions: dict[str, Position] = {}
        pending: list[dict] = []
        trades, curve, vetoes = [], [], {}
        halted = False
        exposure_bars = 0

        def veto(reason):
            vetoes[reason] = vetoes.get(reason, 0) + 1

        def close(pos: Position, px: float, t_exit: int, reason: str):
            nonlocal cash
            exit_px = px * (1 - pos.side * self.adverse)
            fee = pos.qty * exit_px * self.exit_fee
            pnl = pos.side * pos.qty * (exit_px - pos.entry_px)
            if reason == "liquidation":
                pnl = -pos.margin  # isolated margin lost
            cash += pnl - fee  # funding was booked to cash when it settled
            trades.append({"symbol": pos.symbol, "side": "LONG" if pos.side > 0 else "SHORT",
                           "entry_time": pos.entry_time, "exit_time": t_exit, "entry_px": pos.entry_px,
                           "exit_px": exit_px, "qty": pos.qty, "notional": pos.qty * pos.entry_px,
                           "gross_pnl": pnl, "fees": pos.fees + fee, "funding": pos.funding,
                           "net_pnl": pnl - pos.fees - fee - pos.funding, "reason": reason,
                           "hold_bars": int((t_exit - pos.entry_time) // self.step), "regime": pos.regime,
                           "confidence": pos.confidence})

        for t in timeline:
            # 1. fills at this bar's open
            for od in pending:
                b = self.bars[od["symbol"]]
                i = b.idx(t)
                if i < 0 or i >= len(b) or not np.isfinite(b.open[i]):
                    veto("no_bar_at_entry")
                    continue
                px = b.open[i] * (1 + od["side"] * self.adverse)
                tp, sl = od["tp"], od["sl"]
                equity_now = cash  # realised equity: sizing never counts open profits
                notional = min(equity_now * cfg.risk_per_trade / (sl + 2 * self.adverse + self.entry_fee + self.exit_fee),
                               equity_now * cfg.max_position_notional_frac * cfg.max_leverage)
                qty, freason = self._filters_ok(od["symbol"], notional / px, px)
                if freason:
                    veto(freason)
                    continue
                margin = qty * px / cfg.max_leverage
                used = sum(p.margin for p in positions.values())
                if used + margin > cash:
                    veto("insufficient_margin")
                    continue
                fee = qty * px * self.entry_fee
                cash -= fee
                lev = cfg.max_leverage
                liq = px * (1 - od["side"] * (1 / lev - cfg.maintenance_margin_rate))
                positions[od["symbol"]] = Position(
                    od["symbol"], od["side"], qty, px, int(t), i, px * (1 + od["side"] * tp), px * (1 - od["side"] * sl),
                    liq, margin, i + self.H - 1, fee, od["regime"], od["confidence"])
            pending = []

            # 2. exits within this bar
            for sym in list(positions):
                pos = positions[sym]
                b = self.bars[sym]
                i = b.idx(t)
                if i >= len(b) or not np.isfinite(b.high[i]):
                    if i > pos.max_exit_idx:  # data gap / delisting past the holding limit: last known price
                        known = np.nonzero(np.isfinite(b.close[: min(i, len(b))]))[0]
                        close(pos, b.close[known[-1]] if len(known) else pos.entry_px, int(t), "data_gap_exit")
                        del positions[sym]
                    continue
                o, h, lo, c = b.open[i], b.high[i], b.low[i], b.close[i]
                s = pos.side
                adverse_extreme = lo if s > 0 else h
                fav_extreme = h if s > 0 else lo
                if (s > 0 and o <= pos.liq_px) or (s < 0 and o >= pos.liq_px):
                    close(pos, o, int(t), "liquidation")
                elif (s > 0 and o <= pos.sl_px) or (s < 0 and o >= pos.sl_px):
                    close(pos, o, int(t), "stop_gap")
                elif (s > 0 and adverse_extreme <= pos.liq_px) or (s < 0 and adverse_extreme >= pos.liq_px):
                    if (s > 0 and pos.sl_px > pos.liq_px) or (s < 0 and pos.sl_px < pos.liq_px):
                        close(pos, pos.sl_px, int(t), "stop")
                    else:
                        close(pos, pos.liq_px, int(t), "liquidation")
                elif (s > 0 and adverse_extreme <= pos.sl_px) or (s < 0 and adverse_extreme >= pos.sl_px):
                    close(pos, pos.sl_px, int(t), "stop")  # also covers "both touched": stop assumed
                elif (s > 0 and o >= pos.tp_px) or (s < 0 and o <= pos.tp_px):
                    close(pos, o, int(t), "take_profit_gap")
                elif (s > 0 and fav_extreme >= pos.tp_px) or (s < 0 and fav_extreme <= pos.tp_px):
                    close(pos, pos.tp_px, int(t), "take_profit")
                elif i >= pos.max_exit_idx:
                    close(pos, c, int(t) + self.step, "timeout")
                else:
                    continue
                del positions[sym]

            # 3. funding settled during this bar (t, t+step]
            for pos in positions.values():
                b = self.bars[pos.symbol]
                if len(b.funding_t):
                    a_, z_ = np.searchsorted(b.funding_t, [t, t + self.step], side="right")
                    if z_ > a_:
                        i = b.idx(t)
                        mark = b.close[i] if 0 <= i < len(b) and np.isfinite(b.close[i]) else pos.entry_px
                        pay = pos.side * pos.qty * mark * b.funding_r[a_:z_].sum()
                        pos.funding += pay
                        cash -= pay

            # mark to market at this bar's close
            unreal = 0.0
            for pos in positions.values():
                b = self.bars[pos.symbol]
                i = b.idx(t)
                c = b.close[i] if 0 <= i < len(b) and np.isfinite(b.close[i]) else pos.entry_px
                unreal += pos.side * pos.qty * (c - pos.entry_px)
            equity = cash + unreal
            peak = max(peak, equity)
            dd = 1 - equity / peak
            if dd >= cfg.max_portfolio_drawdown:
                halted = True
            exposure_bars += bool(positions)
            curve.append((int(t) + self.step, equity, dd, len(positions)))

            # 4. decisions at this bar's close -> fill at the next open
            group = by_time.get(int(t) + self.step)
            if group is None:
                continue
            for r in group.itertuples(index=False):
                if r.pred not in ("LONG", "SHORT"):
                    continue
                if halted:
                    veto("drawdown_halt")
                    continue
                if r.confidence < cfg.min_confidence:
                    veto("low_confidence")
                    continue
                if self.veto_abnormal and getattr(r, "regime_abnormal", 0) == 1:
                    veto("abnormal_regime")
                    continue
                if cfg.one_position_per_symbol and (r.symbol in positions or any(o["symbol"] == r.symbol for o in pending)):
                    veto("already_in_symbol")
                    continue
                if len(positions) + len(pending) >= cfg.max_positions:
                    veto("max_positions")
                    continue
                b = self.bars.get(r.symbol)
                if b is None:
                    veto("no_market_data")
                    continue
                i = b.idx(int(t))
                if i < 0 or i >= len(b) or not np.isfinite(b.atr_pct[i]):
                    veto("no_atr")
                    continue
                tp, sl = barrier_widths(np.array([b.atr_pct[i]]), self.lcfg)
                pending.append({"symbol": r.symbol, "side": 1 if r.pred == "LONG" else -1, "tp": float(tp[0]),
                                "sl": float(sl[0]), "regime": getattr(r, "regime_trend", ""),
                                "confidence": float(r.confidence)})
        for pos in list(positions.values()):  # end of test window
            b = self.bars[pos.symbol]
            last = int(np.nonzero(np.isfinite(b.close))[0].max())
            close(pos, b.close[last], int(timeline[-1]), "end_of_test")
        return {"trades": pd.DataFrame(trades), "equity": pd.DataFrame(curve, columns=["time", "equity", "drawdown",
                                                                                         "open_positions"]),
                "vetoes": vetoes, "exposure": exposure_bars / max(1, len(timeline))}


def performance(result: dict, initial: float) -> dict:
    tr, eq = result["trades"], result["equity"]
    out = {"initial_equity": initial, "final_equity": float(eq["equity"].iloc[-1]) if len(eq) else initial,
           "trades": int(len(tr)), "vetoes": result["vetoes"], "exposure": result["exposure"]}
    out["net_pnl"] = out["final_equity"] - initial
    out["return_pct"] = 100 * out["net_pnl"] / initial
    out["max_drawdown_pct"] = 100 * float(eq["drawdown"].max()) if len(eq) else 0.0
    if len(eq) > 1:
        daily = eq.assign(day=pd.to_datetime(eq["time"], unit="ms", utc=True).dt.floor("D")).groupby("day")["equity"].last()
        rets = daily.pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            out["sharpe_daily_annualised"] = float(rets.mean() / rets.std() * np.sqrt(365))
            downside = rets[rets < 0].std()
            out["sortino_daily_annualised"] = float(rets.mean() / downside * np.sqrt(365)) if downside > 0 else None
        out["days"] = int(len(daily))
    if len(tr):
        w, l_ = tr[tr["net_pnl"] > 0], tr[tr["net_pnl"] <= 0]
        out.update({
            "win_rate": float(len(w) / len(tr)), "avg_win": float(w["net_pnl"].mean()) if len(w) else None,
            "avg_loss": float(l_["net_pnl"].mean()) if len(l_) else None,
            "profit_factor": float(w["net_pnl"].sum() / -l_["net_pnl"].sum()) if len(l_) and l_["net_pnl"].sum() < 0 else None,
            "expectancy_usdt": float(tr["net_pnl"].mean()), "fees_usdt": float(tr["fees"].sum()),
            "funding_usdt": float(tr["funding"].sum()), "avg_hold_bars": float(tr["hold_bars"].mean()),
            "turnover": float(tr["notional"].sum() * 2 / initial),
            "exit_reasons": tr["reason"].value_counts().to_dict(),
            "by_symbol": tr.groupby("symbol")["net_pnl"].agg(["count", "sum"]).round(4).reset_index().to_dict("records"),
            "by_regime": tr.groupby("regime")["net_pnl"].agg(["count", "sum"]).round(4).reset_index().to_dict("records"),
            "by_side": tr.groupby("side")["net_pnl"].agg(["count", "sum"]).round(4).reset_index().to_dict("records"),
        })
    return out
