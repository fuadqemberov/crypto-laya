"""Market-structure and Smart Money Concepts features with explicit, causal definitions.

Definitions (L = `swing_lookback`):

- **Swing high** at pivot bar p: `high[p]` is the strict maximum of `high[p-L .. p+L]`. It can only
  be known once bar p+L has closed, so it is *confirmed* (and usable) at bar p+L, never at p.
  Swing lows mirror this with `low`.
- **HH / HL / LH / LL**: each newly confirmed swing high is compared with the previous confirmed
  swing high (higher high or lower high); likewise for lows.
- **Break of structure (BOS) / change of character (CHoCH)**: bar t closes above the latest
  confirmed swing high (having closed at or below it on t-1) is an up-break; a close below the latest
  confirmed swing low is a down-break. A break in the direction of the prevailing structural trend
  (the direction of the last break) is a BOS; a break against it is a CHoCH. The first break sets
  the direction and counts as a BOS.
- **Fair value gap (FVG)**: bullish at bar t when `low[t] > high[t-2]` and the gap is at least
  `fvg_min_atr` x ATR; bearish when `high[t] < low[t-2]`. Known at t's close.
- **Order block**: at an up-break (BOS or CHoCH), the bullish order block is the last bar with
  `close < open` within the previous `order_block_lookback` bars; its [low, high] is the zone.
  Bearish order blocks mirror this at down-breaks.
- **Liquidity sweep**: bar t trades above the highest high of the previous `sweep_lookback` bars
  but closes back below it (sweep of buy-side liquidity); mirror for lows.
- **Support / resistance**: highest high and lowest low of the last `sr_window` bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def confirmed_swings(high: np.ndarray, low: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays marking bars t at which a swing (pivot at t-L) becomes confirmed."""
    L = lookback
    h, lo = pd.Series(high), pd.Series(low)
    # strictly above the L bars before the pivot and the L bars after it (bars p+1..t, t = p+L)
    left_max = h.shift(1).rolling(L, min_periods=L).max()
    right_max = h.rolling(L, min_periods=L).max()
    left_min = lo.shift(1).rolling(L, min_periods=L).min()
    right_min = lo.rolling(L, min_periods=L).min()
    pivot_h, pivot_l = h.shift(L), lo.shift(L)
    sh = ((pivot_h > left_max.shift(L)) & (pivot_h > right_max)).to_numpy()
    sl = ((pivot_l < left_min.shift(L)) & (pivot_l < right_min)).to_numpy()
    return sh, sl


def structure_features(df: pd.DataFrame, atr_series: pd.Series, lookback: int = 3, fvg_min_atr: float = 0.1,
                       ob_lookback: int = 20, sweep_lookback: int = 48, sr_window: int = 288) -> pd.DataFrame:
    o, h, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    a = atr_series.to_numpy(dtype=float)
    n = len(df)
    sh_conf, sl_conf = confirmed_swings(h, lo, lookback)

    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    hh = np.zeros(n)  # +1 higher high, -1 lower high (state of the latest swing-high comparison)
    hl = np.zeros(n)  # +1 higher low,  -1 lower low
    bos_up = np.zeros(n, dtype=np.int8)
    bos_dn = np.zeros(n, dtype=np.int8)
    choch_up = np.zeros(n, dtype=np.int8)
    choch_dn = np.zeros(n, dtype=np.int8)
    struct_dir = np.zeros(n)
    ob_bull_lo = np.full(n, np.nan)
    ob_bull_hi = np.full(n, np.nan)
    ob_bear_lo = np.full(n, np.nan)
    ob_bear_hi = np.full(n, np.nan)

    cur_sh = cur_sl = np.nan
    prev_sh = prev_sl = np.nan
    hh_state = hl_state = 0.0
    direction = 0.0
    obl_lo = obl_hi = obr_lo = obr_hi = np.nan
    for t in range(n):
        if sh_conf[t]:
            prev_sh, cur_sh = cur_sh, h[t - lookback]
            if not np.isnan(prev_sh):
                hh_state = 1.0 if cur_sh > prev_sh else -1.0
        if sl_conf[t]:
            prev_sl, cur_sl = cur_sl, lo[t - lookback]
            if not np.isnan(prev_sl):
                hl_state = 1.0 if cur_sl > prev_sl else -1.0
        if t > 0 and not np.isnan(cur_sh) and c[t] > cur_sh >= c[t - 1]:
            if direction < 0:
                choch_up[t] = 1
            else:
                bos_up[t] = 1
            direction = 1.0
            for j in range(t - 1, max(-1, t - 1 - ob_lookback), -1):
                if c[j] < o[j]:
                    obl_lo, obl_hi = lo[j], h[j]
                    break
        if t > 0 and not np.isnan(cur_sl) and c[t] < cur_sl <= c[t - 1]:
            if direction > 0:
                choch_dn[t] = 1
            else:
                bos_dn[t] = 1
            direction = -1.0
            for j in range(t - 1, max(-1, t - 1 - ob_lookback), -1):
                if c[j] > o[j]:
                    obr_lo, obr_hi = lo[j], h[j]
                    break
        last_sh[t], last_sl[t] = cur_sh, cur_sl
        hh[t], hl[t], struct_dir[t] = hh_state, hl_state, direction
        ob_bull_lo[t], ob_bull_hi[t], ob_bear_lo[t], ob_bear_hi[t] = obl_lo, obl_hi, obr_lo, obr_hi

    out = pd.DataFrame(index=df.index)
    safe_a = np.where(a > 0, a, np.nan)
    out["swing_high_dist_atr"] = (last_sh - c) / safe_a
    out["swing_low_dist_atr"] = (c - last_sl) / safe_a
    out["hh_lh"] = hh
    out["hl_ll"] = hl
    out["structure_dir"] = struct_dir
    out["bos_up"], out["bos_down"] = bos_up, bos_dn
    out["choch_up"], out["choch_down"] = choch_up, choch_dn
    out["bars_since_break"] = _bars_since(bos_up | bos_dn | choch_up | choch_dn)

    gap_bull = np.r_[np.full(2, np.nan), lo[2:] - h[:-2]]
    gap_bear = np.r_[np.full(2, np.nan), lo[:-2] - h[2:]]
    fvg_bull = (gap_bull > 0) & (gap_bull >= fvg_min_atr * a)
    fvg_bear = (gap_bear > 0) & (gap_bear >= fvg_min_atr * a)
    out["fvg_bull"] = fvg_bull.astype(np.int8)
    out["fvg_bear"] = fvg_bear.astype(np.int8)
    out["fvg_bull_size_atr"] = np.where(fvg_bull, gap_bull / safe_a, 0.0)
    out["fvg_bear_size_atr"] = np.where(fvg_bear, gap_bear / safe_a, 0.0)
    out["bars_since_fvg_bull"] = _bars_since(fvg_bull)
    out["bars_since_fvg_bear"] = _bars_since(fvg_bear)

    out["ob_bull_dist_atr"] = (c - ob_bull_hi) / safe_a
    out["in_ob_bull"] = ((c >= ob_bull_lo) & (c <= ob_bull_hi)).astype(np.int8)
    out["ob_bear_dist_atr"] = (ob_bear_lo - c) / safe_a
    out["in_ob_bear"] = ((c >= ob_bear_lo) & (c <= ob_bear_hi)).astype(np.int8)

    prior_hi = pd.Series(h).shift(1).rolling(sweep_lookback, min_periods=sweep_lookback).max().to_numpy()
    prior_lo = pd.Series(lo).shift(1).rolling(sweep_lookback, min_periods=sweep_lookback).min().to_numpy()
    out["sweep_high"] = ((h > prior_hi) & (c < prior_hi)).astype(np.int8)
    out["sweep_low"] = ((lo < prior_lo) & (c > prior_lo)).astype(np.int8)

    res = pd.Series(h).rolling(sr_window, min_periods=sr_window // 4).max().to_numpy()
    sup = pd.Series(lo).rolling(sr_window, min_periods=sr_window // 4).min().to_numpy()
    out["resistance_dist_atr"] = (res - c) / safe_a
    out["support_dist_atr"] = (c - sup) / safe_a
    return out


def _bars_since(flag: np.ndarray, cap: int = 10_000) -> np.ndarray:
    out = np.full(len(flag), np.nan)
    last = -1
    for i, f in enumerate(flag):
        if f:
            last = i
        if last >= 0:
            out[i] = min(i - last, cap)
    return out
