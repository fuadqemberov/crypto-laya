"""Cost-aware triple-barrier labels from future price paths.

Execution model, for a decision at the close of bar t:

- **Entry** at the open of bar t+1 (the first price that exists after the decision).
- **Barriers** as fractions of the entry price: `tp = tp_atr * ATR%_t`, `sl = sl_atr * ATR%_t`
  (ATR% known at t), scaled up together if the take-profit would fall below `min_barrier_pct`.
- **Path**: bars t+1 .. t+H. A barrier is touched when a bar's high/low reaches it.
- **Same-candle ambiguity**: if one bar touches both barriers the order is unknown from OHLC.
  `conservative_sl` counts it as the stop, `resolve_1m` looks inside that bar at 1m candles
  (still ambiguous inside one minute -> stop), `drop` makes the label missing. TP-first is never
  assumed.
- **Vertical barrier**: no touch by t+H exits at the close of bar t+H.
- **Costs**: round-trip fees + spread + slippage, plus funding settled in (entry, exit].
- Any missing bar in the path makes the label missing (unreliable timestamps are not guessed).

Labels: LONG if the long trade reaches its take-profit first and nets a positive return after
costs; SHORT likewise; WAIT otherwise. With `tp >= sl` LONG and SHORT are mutually exclusive.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config import LabelingConfig
from ..utils.timeframes import horizon_bars, tf_ms

CLASSES = ("LONG", "SHORT", "WAIT")
OUTCOME_TP, OUTCOME_SL, OUTCOME_TIMEOUT = 1, -1, 0


def _forward_matrix(x: np.ndarray, H: int) -> np.ndarray:
    """M[t, k-1] = x[t+k] for k in 1..H (NaN past the end)."""
    n = len(x)
    pad = np.concatenate([x, np.full(H + 1, np.nan)])
    idx = np.arange(n)[:, None] + np.arange(1, H + 1)[None, :]
    return pad[idx]


def _first_true(m: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or -1."""
    any_ = m.any(axis=1)
    return np.where(any_, m.argmax(axis=1), -1)


def barrier_widths(atr_pct: np.ndarray, cfg: LabelingConfig, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if cfg.method == "triple_barrier_pct":
        tp = np.full_like(atr_pct, cfg.tp_pct * scale, dtype=float)
        sl = np.full_like(atr_pct, cfg.sl_pct * scale, dtype=float)
    else:
        tp = cfg.tp_atr * atr_pct * scale
        sl = cfg.sl_atr * atr_pct * scale
    with np.errstate(invalid="ignore", divide="ignore"):
        boost = np.where(tp < cfg.min_barrier_pct, cfg.min_barrier_pct / tp, 1.0)
    return tp * boost, sl * boost


def _side_outcome(hi: np.ndarray, lo: np.ndarray, entry: np.ndarray, tp: np.ndarray, sl: np.ndarray,
                  side: int, ambiguity: str, resolver=None, open_time_path: np.ndarray | None = None):
    """Outcome, exit offset (1..H) and exit price for one side. side=+1 long, -1 short."""
    if side > 0:
        tp_px, sl_px = entry * (1 + tp), entry * (1 - sl)
        hit_tp = hi >= tp_px[:, None]
        hit_sl = lo <= sl_px[:, None]
    else:
        tp_px, sl_px = entry * (1 - tp), entry * (1 + sl)
        hit_tp = lo <= tp_px[:, None]
        hit_sl = hi >= sl_px[:, None]
    ftp, fsl = _first_true(hit_tp), _first_true(hit_sl)
    n, H = hi.shape
    outcome = np.full(n, OUTCOME_TIMEOUT)
    exit_k = np.full(n, H - 1)
    tp_first = (ftp >= 0) & ((fsl < 0) | (ftp < fsl))
    sl_first = (fsl >= 0) & ((ftp < 0) | (fsl < ftp))
    same = (ftp >= 0) & (ftp == fsl)
    outcome[tp_first], exit_k[tp_first] = OUTCOME_TP, ftp[tp_first]
    outcome[sl_first], exit_k[sl_first] = OUTCOME_SL, fsl[sl_first]
    ambiguous = same.copy()
    valid = np.ones(n, dtype=bool)
    if same.any():
        exit_k[same] = ftp[same]
        if ambiguity == "drop":
            valid[same] = False
        elif ambiguity == "resolve_1m" and resolver is not None and open_time_path is not None:
            for i in np.nonzero(same)[0]:
                bar_open = open_time_path[i, ftp[i]]
                res = resolver(int(bar_open), tp_px[i], sl_px[i], side)
                outcome[i] = res
                ambiguous[i] = res == OUTCOME_SL and resolver.last_was_ambiguous
        else:
            outcome[same] = OUTCOME_SL
    exit_px = np.where(outcome == OUTCOME_TP, tp_px, np.where(outcome == OUTCOME_SL, sl_px, np.nan))
    return outcome, exit_k, exit_px, ambiguous, valid


class OneMinuteResolver:
    """Orders TP vs SL inside one base bar using its 1m candles; still-ambiguous minutes -> SL."""

    def __init__(self, one_min: pd.DataFrame, base_ms: int):
        self.df = one_min.set_index("open_time")[["high", "low"]]
        self.base_ms = base_ms
        self.last_was_ambiguous = False

    def __call__(self, bar_open: int, tp_px: float, sl_px: float, side: int) -> int:
        sub = self.df.loc[bar_open: bar_open + self.base_ms - 60_000]
        self.last_was_ambiguous = True
        if len(sub) * 60_000 != self.base_ms:
            return OUTCOME_SL
        for hi, lo in sub.itertuples(index=False):
            tp_hit = hi >= tp_px if side > 0 else lo <= tp_px
            sl_hit = lo <= sl_px if side > 0 else hi >= sl_px
            if tp_hit and sl_hit:
                return OUTCOME_SL
            if tp_hit:
                self.last_was_ambiguous = False
                return OUTCOME_TP
            if sl_hit:
                self.last_was_ambiguous = False
                return OUTCOME_SL
        return OUTCOME_SL


def funding_between(funding: pd.DataFrame | None, start_ms: np.ndarray, end_ms: np.ndarray) -> np.ndarray:
    """Sum of funding rates settled in (start, end] for each pair."""
    if funding is None or len(funding) == 0:
        return np.zeros(len(start_ms))
    f = funding.sort_values("funding_time")
    t = f["funding_time"].to_numpy(dtype="int64")
    cs = np.concatenate([[0.0], np.cumsum(f["funding_rate"].to_numpy(dtype=float))])
    a = np.searchsorted(t, start_ms, side="right")
    b = np.searchsorted(t, end_ms, side="right")
    return cs[b] - cs[a]


def label_frame(df: pd.DataFrame, atr_pct: pd.Series | np.ndarray, cfg: LabelingConfig, timeframe: str = "5m",
                funding: pd.DataFrame | None = None, one_minute: pd.DataFrame | None = None,
                scale: float = 1.0) -> pd.DataFrame:
    """Labels for every row of a full-grid candle frame (missing bars = NaN rows)."""
    step = tf_ms(timeframe)
    H = cfg.max_hold_bars
    o = df["open"].to_numpy(dtype=float)
    hi_m = _forward_matrix(df["high"].to_numpy(dtype=float), H)
    lo_m = _forward_matrix(df["low"].to_numpy(dtype=float), H)
    cl_m = _forward_matrix(df["close"].to_numpy(dtype=float), H)
    ot = df["open_time"].to_numpy(dtype="int64")
    ot_m = _forward_matrix(ot.astype(float), H)
    entry = np.concatenate([o[1:], [np.nan]])
    atr_pct = np.asarray(atr_pct, dtype=float)
    tp, sl = barrier_widths(atr_pct, cfg, scale)
    path_ok = np.isfinite(hi_m).all(1) & np.isfinite(lo_m).all(1) & np.isfinite(cl_m).all(1) & np.isfinite(entry)
    ok = path_ok & np.isfinite(tp) & np.isfinite(sl) & (tp > 0)
    resolver = OneMinuteResolver(one_minute, step) if (cfg.ambiguity == "resolve_1m" and one_minute is not None) else None
    safe_entry = np.where(ok, entry, 1.0)
    out = {}
    costs = cfg.costs.round_trip()
    entry_time = ot + step
    for side, name in ((1, "long"), (-1, "short")):
        outcome, k, exit_px, amb, valid = _side_outcome(np.where(ok[:, None], hi_m, 0), np.where(ok[:, None], lo_m, 0),
                                                        safe_entry, np.where(ok, tp, 0.01), np.where(ok, sl, 0.01),
                                                        side, cfg.ambiguity, resolver, ot_m)
        timeout_px = cl_m[:, H - 1]
        exit_px = np.where(outcome == OUTCOME_TIMEOUT, timeout_px, exit_px)
        gross = side * (exit_px / safe_entry - 1)
        exit_time = np.nan_to_num(ot_m[np.arange(len(ot)), k], nan=0).astype("int64") + step
        fund = funding_between(funding, entry_time, exit_time) if cfg.costs.include_funding else 0.0
        net = gross - costs - side * fund
        valid &= ok
        out[f"{name}_outcome"] = np.where(valid, outcome, np.nan)
        out[f"{name}_exit_bars"] = np.where(valid, k + 1, np.nan)
        out[f"{name}_net_ret"] = np.where(valid, net, np.nan)
        out[f"{name}_ambiguous"] = np.where(valid, amb, np.nan)
        out[f"{name}_tp_first"] = np.where(valid, (outcome == OUTCOME_TP).astype(float), np.nan)
        out[f"{name}_sl_first"] = np.where(valid, (outcome == OUTCOME_SL).astype(float), np.nan)
    res = pd.DataFrame(out, index=df.index)
    long_win = (res["long_outcome"] == OUTCOME_TP) & (res["long_net_ret"] > 0)
    short_win = (res["short_outcome"] == OUTCOME_TP) & (res["short_net_ret"] > 0)
    if cfg.timeout_policy == "sign_if_net_positive":
        long_win |= (res["long_outcome"] == OUTCOME_TIMEOUT) & (res["long_net_ret"] > 0)
        short_win |= (res["short_outcome"] == OUTCOME_TIMEOUT) & (res["short_net_ret"] > 0)
    label = np.select([long_win & ~short_win, short_win & ~long_win], ["LONG", "SHORT"], "WAIT").astype(object)
    valid_any = res["long_outcome"].notna() & res["short_outcome"].notna()
    label[~valid_any.to_numpy()] = None
    res["label"] = label
    res["tp_pct"], res["sl_pct"] = np.where(ok, tp, np.nan), np.where(ok, sl, np.nan)
    res["entry_price"] = np.where(ok, entry, np.nan)
    # horizon end (for purging / embargo): close of bar t+H
    res["label_end_time"] = ot + (H + 1) * step
    # regression targets and path risk
    for minutes in cfg.return_horizons_min:
        hb = horizon_bars(minutes, timeframe)
        if hb <= H:
            res[f"fwd_ret_{minutes}m"] = np.where(path_ok, np.log(cl_m[:, hb - 1] / safe_entry), np.nan)
        else:
            c_fwd = df["close"].shift(-hb).to_numpy(dtype=float)
            res[f"fwd_ret_{minutes}m"] = np.log(c_fwd / entry)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are masked by path_ok
        res["mfe"] = np.where(path_ok, np.nanmax(hi_m, axis=1) / safe_entry - 1, np.nan)
        res["mae"] = np.where(path_ok, np.nanmin(lo_m, axis=1) / safe_entry - 1, np.nan)
    return res


def soft_label_distribution(df: pd.DataFrame, atr_pct, cfg: LabelingConfig, timeframe: str = "5m",
                            funding: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of barrier scales (e.g. 0.75x / 1x / 1.25x) that give each class: a soft target that
    expresses how sensitive the label is to the exact barrier choice."""
    counts = {c: np.zeros(len(df)) for c in CLASSES}
    valid = np.zeros(len(df))
    for s in cfg.soft_barrier_scales:
        lab = label_frame(df, atr_pct, cfg, timeframe, funding, scale=s)["label"].to_numpy()
        for c in CLASSES:
            counts[c] += lab == c
        valid += pd.notna(lab)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({f"p_{c}": np.where(valid == len(cfg.soft_barrier_scales), counts[c] / valid, np.nan)
                             for c in CLASSES}, index=df.index)
