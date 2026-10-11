"""Deterministic market-state text for LAYA, and the LONG/SHORT/WAIT question.

The same row always serialises to the same string (fixed key order, fixed precision), so
training, evaluation and inference all see one representation. Missing values are written as
`NA`, never filled in.
"""
from __future__ import annotations

import math

import pandas as pd

QUESTION_ID = "decision"
QUESTION = {
    "type": "choice",
    "instructions": ("Given this crypto perpetual-futures market state at a candle close, decide the action "
                     "for the next bars. LONG and SHORT require the move to reach its take-profit before its "
                     "stop after trading costs; otherwise WAIT."),
    "criteria": {
        "LONG": "open a long: price likely reaches the upside target before the stop",
        "SHORT": "open a short: price likely reaches the downside target before the stop",
        "WAIT": "stay flat: no edge after costs, or the outcome is unclear",
    },
}

# (state key, feature column, decimals). Kept compact to fit LAYA's token budget.
STATE_FIELDS: list[tuple[str, str, int]] = [
    ("RSI", "rsi", 1),
    ("MACD_HIST_BP", "macd_hist_pct", 2),
    ("EMA21_DIST_ATR", "ema21_dist_atr", 2),
    ("EMA50_DIST_ATR", "ema50_dist_atr", 2),
    ("EMA200_DIST_ATR", "ema200_dist_atr", 2),
    ("ADX", "adx", 1),
    ("DI_SPREAD", "di_spread", 1),
    ("RET_1H_PCT", "ret_12", 2),
    ("RET_4H_PCT", "ret_48", 2),
    ("STOCH_RSI_K", "stoch_rsi_k", 2),
    ("ATR_PCT", "atr_pct", 3),
    ("BB_WIDTH", "bb_width", 3),
    ("VOL_PCTL", "vol_percentile", 2),
    ("REL_VOLUME", "rel_volume", 2),
    ("TAKER_IMB_1H", "taker_imbalance_12", 3),
    ("CVD_1D", "cvd_288_norm", 3),
    ("VWAP_DIST_ATR", "vwap_dist_atr", 2),
    ("STRUCTURE", "structure_dir", 0),
    ("HH_LH", "hh_lh", 0),
    ("HL_LL", "hl_ll", 0),
    ("BARS_SINCE_BREAK", "bars_since_break", 0),
    ("SWEEP_HIGH", "sweep_high", 0),
    ("SWEEP_LOW", "sweep_low", 0),
    ("RES_DIST_ATR", "resistance_dist_atr", 2),
    ("SUP_DIST_ATR", "support_dist_atr", 2),
    ("FUNDING_BP", "funding_rate", 2),
    ("OI_CHG_1H_PCT", "oi_change_12", 2),
    ("PREMIUM_BP", "mark_index_premium", 2),
    ("TREND_1H", "htf_1h_trend", 0),
    ("RSI_1H", "htf_1h_rsi", 1),
    ("TREND_4H", "htf_4h_trend", 0),
    ("RSI_4H", "htf_4h_rsi", 1),
    ("BTC_RET_1H_PCT", "ctx_btc_ret_12", 2),
    ("ETH_RET_1H_PCT", "ctx_eth_ret_12", 2),
]
# basis points / percent scaling so numbers stay short and comparable across symbols
SCALE = {"macd_hist_pct": 1e4, "funding_rate": 1e4, "mark_index_premium": 1e4, "ret_12": 100, "ret_48": 100,
         "oi_change_12": 100, "ctx_btc_ret_12": 100, "ctx_eth_ret_12": 100, "atr_pct": 100}
TREND_WORD = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}


def _fmt(value, decimals: int) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)) or pd.isna(value):
        return "NA"
    v = float(value)
    if decimals == 0:
        return str(int(round(v)))
    s = f"{v:.{decimals}f}"
    return "0" if float(s) == 0 else s


def serialize_row(row: pd.Series | dict, symbol: str, timeframe: str, fields=None) -> str:
    """`KEY=VALUE` lines in a fixed order. Symbol and timeframe go first."""
    fields = fields or STATE_FIELDS
    get = row.get
    parts = [f"SYMBOL={symbol}", f"TIMEFRAME={timeframe}"]
    for key, col, dec in fields:
        v = get(col)
        if key.startswith("TREND_") and v is not None and not pd.isna(v):
            parts.append(f"{key}={TREND_WORD.get(int(v), 'NA')}")
            continue
        if v is not None and not pd.isna(v) and col in SCALE:
            v = float(v) * SCALE[col]
        parts.append(f"{key}={_fmt(v, dec)}")
    rt, rv = get("regime_trend"), get("regime_vol")
    parts.append(f"REGIME={rt if isinstance(rt, str) else 'NA'}/{rv if isinstance(rv, str) else 'NA'}")
    if get("regime_abnormal") == 1:
        parts.append("ABNORMAL=1")
    return "\n".join(parts)


def laya_row(state: str, label: str | None = None, soft: dict[str, float] | None = None) -> dict:
    """One `{state, questions, expected|gold}` row in the schema `laya.train` reads."""
    row = {"state": state, "questions": {QUESTION_ID: QUESTION}}
    if soft is not None:
        row["gold"] = {QUESTION_ID: {"probabilities": soft}}
    elif label is not None:
        row["expected"] = {QUESTION_ID: label}
    return row
