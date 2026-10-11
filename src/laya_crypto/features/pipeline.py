"""Deterministic feature pipeline for one symbol.

The decision time of base bar t is its close (`open_time + timeframe`). Every feature on row t is
computed from candles that closed at or before that time, and every external series (funding,
open interest, premium, depth, context symbols) is joined *as of* its own availability time.
Missing inputs stay NaN and get an explicit `has_*` indicator: nothing is imputed with invented
market data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import FeaturesConfig
from ..utils.timeframes import tf_ms
from . import indicators as ind
from . import mtf
from .regime import rule_regime
from .structure import structure_features

BASE_COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "count",
             "taker_buy_volume", "taker_buy_quote_volume"]


@dataclass
class AuxData:
    funding: pd.DataFrame | None = None  # funding_time, funding_rate, funding_interval_hours
    metrics: pd.DataFrame | None = None  # time, sum_open_interest, ...
    premium: pd.DataFrame | None = None  # premium index klines (open_time, close, ...)
    mark: pd.DataFrame | None = None
    index: pd.DataFrame | None = None
    flow: pd.DataFrame | None = None  # 1m order-flow bars from aggTrades
    depth: pd.DataFrame | None = None  # depth snapshots
    metrics_lag_ms: int = 300_000


@dataclass
class FeatureFrame:
    frame: pd.DataFrame
    groups: dict[str, list[str]] = field(default_factory=dict)

    def columns(self, groups: list[str] | None = None) -> list[str]:
        gs = groups or list(self.groups)
        return [c for g in gs for c in self.groups.get(g, [])]


def asof(decision_time: pd.Series, right: pd.DataFrame, time_col: str, cols: list[str],
         lag_ms: int = 0, max_age_ms: int | None = None) -> pd.DataFrame:
    """Values of `cols` from the latest `right` row with time_col + lag_ms <= decision_time."""
    r = right[[time_col, *cols]].dropna(subset=[time_col]).copy()
    r["_avail"] = r[time_col].astype("int64") + lag_ms
    r = r.sort_values("_avail").drop(columns=[time_col])
    left = pd.DataFrame({"_t": decision_time.astype("int64").to_numpy(), "_i": np.arange(len(decision_time))})
    m = pd.merge_asof(left.sort_values("_t"), r, left_on="_t", right_on="_avail", direction="backward",
                      tolerance=max_age_ms)
    m = m.sort_values("_i")
    out = m[cols].set_axis(decision_time.index)
    out["_age_ms"] = (m["_t"] - m["_avail"]).to_numpy()
    return out


def build_features(base: pd.DataFrame, cfg: FeaturesConfig, base_tf: str = "5m", aux: AuxData | None = None,
                   context: dict[str, pd.DataFrame] | None = None) -> FeatureFrame:
    step = tf_ms(base_tf)
    df = base[[c for c in BASE_COLS if c in base.columns]].sort_values("open_time")
    # Full bar grid: a gap becomes NaN rows, so no rolling window silently spans missing time.
    grid = np.arange(df["open_time"].iloc[0], df["open_time"].iloc[-1] + step, step, dtype="int64")
    df = df.set_index("open_time").reindex(grid).rename_axis("open_time").reset_index()
    df["bar_present"] = df["close"].notna()
    aux = aux or AuxData()
    df["decision_time"] = df["open_time"] + step
    h, lo, c, v = df["high"], df["low"], df["close"], df["volume"]
    groups: dict[str, list[str]] = {}
    feats: dict[str, pd.Series] = {}

    def add(group: str, name: str, values) -> None:
        feats[name] = pd.Series(np.asarray(values, dtype=float) if not isinstance(values, pd.Series) else values,
                                index=df.index)
        groups.setdefault(group, []).append(name)

    atr = ind.atr(h, lo, c, cfg.atr_period)
    atr_pct = atr / c

    # ---- trend & momentum
    add("trend", "rsi", ind.rsi(c, cfg.rsi_period))
    m = ind.macd(c, *cfg.macd)
    add("trend", "macd_hist_pct", m["macd_hist"] / c)
    add("trend", "macd_pct", m["macd"] / c)
    emas = {n: ind.ema(c, n) for n in cfg.ema_periods}
    for n, e in emas.items():
        add("trend", f"ema{n}_dist_atr", (c - e) / atr)
    for n in cfg.sma_periods:
        add("trend", f"sma{n}_dist_atr", (c - ind.sma(c, n)) / atr)
    dmi = ind.adx(h, lo, c, cfg.adx_period)
    add("trend", "adx", dmi["adx"])
    add("trend", "di_spread", dmi["plus_di"] - dmi["minus_di"])
    for n in cfg.roc_periods:
        add("trend", f"ret_{n}", ind.roc(c, n))
    sr = ind.stoch_rsi(c, *cfg.stoch_rsi)
    add("trend", "stoch_rsi_k", sr["stoch_rsi_k"])
    add("trend", "stoch_rsi_d", sr["stoch_rsi_d"])
    add("trend", "momentum_12_atr", ind.momentum(c, 12) / atr)
    ema_fast_n, ema_slow_n = sorted(cfg.ema_periods)[1], sorted(cfg.ema_periods)[2]
    ema_slope = ind.slope(emas[ema_fast_n], cfg.slope_window)
    add("trend", f"ema_slope_{ema_fast_n}", ema_slope)
    add("trend", f"ema_slope_{ema_slow_n}", ind.slope(emas[ema_slow_n], cfg.slope_window))
    add("trend", "trend_strength", (dmi["adx"] / 100) * np.sign(emas[ema_fast_n] - emas[ema_slow_n]))

    # ---- volatility
    add("volatility", "atr_pct", atr_pct)
    add("volatility", "true_range_pct", ind.true_range(h, lo, c) / c)
    bb = ind.bollinger(c, cfg.bb_period, cfg.bb_std)
    add("volatility", "bb_width", bb["bb_width"])
    add("volatility", "bb_pctb", bb["bb_pctb"])
    for n in cfg.vol_windows:
        add("volatility", f"realized_vol_{n}", ind.realized_vol(c, n))
    rv_mid = feats[f"realized_vol_{sorted(cfg.vol_windows)[1]}"]
    vol_pct = ind.rolling_percentile(rv_mid, cfg.vol_percentile_window)
    add("volatility", "vol_percentile", vol_pct)
    add("volatility", "range_pct", (h - lo) / c)

    # ---- volume & order flow
    rel = v / v.rolling(cfg.rel_volume_window, min_periods=cfg.rel_volume_window).mean()
    add("volume", "rel_volume", rel)
    vz = ind.zscore(np.log1p(v), cfg.rel_volume_window)
    add("volume", "volume_z", vz)
    add("volume", "volume_spike", (vz >= cfg.volume_spike_z).astype(float).where(vz.notna()))
    if "taker_buy_volume" in df:
        buy = df["taker_buy_volume"]
        delta = 2 * buy - v
        add("volume", "taker_imbalance", delta / v.replace(0, np.nan))
        for n in (12, 48):
            add("volume", f"taker_imbalance_{n}", delta.rolling(n, min_periods=n).sum()
                / v.rolling(n, min_periods=n).sum().replace(0, np.nan))
        # CVD from kline taker volumes: cumulative delta over a window, scaled by volume
        add("volume", "cvd_288_norm", delta.rolling(288, min_periods=288).sum()
            / v.rolling(288, min_periods=288).sum().replace(0, np.nan))
    if "quote_volume" in df:
        vwap = ind.rolling_vwap(c, v, df["quote_volume"], cfg.vwap_window)
        add("volume", "vwap_dist_atr", (c - vwap) / atr)
    if aux.flow is not None and len(aux.flow):
        f = aux.flow.copy()
        f["bucket"] = (f["open_time"] // step) * step
        fb = f.groupby("bucket")[["buy_qty", "sell_qty", "trades"]].sum()
        fb = fb.reindex(df["open_time"].to_numpy())
        has = fb["trades"].notna().to_numpy()
        imb = ((fb["buy_qty"] - fb["sell_qty"]) / (fb["buy_qty"] + fb["sell_qty"])).to_numpy()
        add("volume", "aggr_trade_imbalance", imb)
        add("volume", "has_trade_flow", has.astype(float))
    if aux.depth is not None and len(aux.depth):
        d = asof(df["decision_time"], aux.depth, "time",
                 ["bid_notional_1pct", "ask_notional_1pct"], 0, max_age_ms=3 * step)
        tot = d["bid_notional_1pct"] + d["ask_notional_1pct"]
        add("volume", "depth_imbalance_1pct", (d["bid_notional_1pct"] - d["ask_notional_1pct"]) / tot)
        add("volume", "has_depth", tot.notna().astype(float))

    # ---- market structure / SMC
    st = structure_features(df, atr, cfg.swing_lookback, cfg.fvg_min_atr, cfg.order_block_lookback,
                            cfg.sweep_lookback, cfg.sr_window)
    for col in st.columns:
        add("structure", col, st[col])

    # ---- derivatives context
    if aux.funding is not None and len(aux.funding):
        fr = aux.funding.sort_values("funding_time").copy()
        fr["funding_change"] = fr["funding_rate"].diff()
        j = asof(df["decision_time"], fr, "funding_time", ["funding_rate", "funding_change"], 0,
                 max_age_ms=3 * 8 * 3_600_000)
        add("derivatives", "funding_rate", j["funding_rate"])
        add("derivatives", "funding_change", j["funding_change"])
        add("derivatives", "has_funding", j["funding_rate"].notna().astype(float))
    if aux.metrics is not None and len(aux.metrics):
        mt = aux.metrics.sort_values("time")
        cols = ["sum_open_interest", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
        j = asof(df["decision_time"], mt, "time", [c_ for c_ in cols if c_ in mt], aux.metrics_lag_ms,
                 max_age_ms=4 * step)
        oi = j["sum_open_interest"]
        for n in (1, 12, 48):
            add("derivatives", f"oi_change_{n}", oi / oi.shift(n) - 1)
        price_chg = c / c.shift(12) - 1
        oi_chg = oi / oi.shift(12) - 1
        add("derivatives", "price_oi_divergence", np.sign(price_chg) * -np.sign(oi_chg) * oi_chg.abs())
        # explicit proxies for crowded positioning; not a measure of actual positions
        if "sum_toptrader_long_short_ratio" in j:
            add("derivatives", "proxy_toptrader_ls_ratio", j["sum_toptrader_long_short_ratio"])
        if "count_long_short_ratio" in j:
            add("derivatives", "proxy_account_ls_ratio", j["count_long_short_ratio"])
        if "sum_taker_long_short_vol_ratio" in j:
            add("derivatives", "taker_ls_vol_ratio", j["sum_taker_long_short_vol_ratio"])
        add("derivatives", "has_open_interest", oi.notna().astype(float))
    if aux.mark is not None and aux.index is not None and len(aux.mark) and len(aux.index):
        mk = aux.mark.set_index("open_time")["close"].reindex(df["open_time"].to_numpy()).to_numpy()
        ix = aux.index.set_index("open_time")["close"].reindex(df["open_time"].to_numpy()).to_numpy()
        add("derivatives", "mark_index_premium", mk / ix - 1)
        add("derivatives", "last_mark_basis", c.to_numpy() / mk - 1)
    if aux.premium is not None and len(aux.premium):
        pi = aux.premium.set_index("open_time")["close"].reindex(df["open_time"].to_numpy()).to_numpy()
        add("derivatives", "premium_index", pi)

    # ---- higher timeframes (only complete, already-closed candles)
    for htf in cfg.higher_timeframes:
        hdf = mtf.resample_complete(df, base_tf, htf)
        hf = mtf.htf_features(hdf, f"htf_{htf}", cfg.rsi_period, cfg.atr_period, cfg.adx_period)
        joined = mtf.join_htf(df, hf, base_tf, max_age_ms=2 * tf_ms(htf))
        for col in joined.columns:
            add("mtf", col, joined[col])

    # ---- regime (causal rule-based; the GMM detector is fitted on the train split later)
    ret_z = ind.zscore(ind.log_returns(c), 288)
    reg = rule_regime(dmi["adx"], emas[ema_fast_n], emas[ema_slow_n], ema_slope, vol_pct, ret_z,
                      cfg.regime_trend_adx, cfg.regime_vol_high_pct, cfg.regime_vol_low_pct,
                      cfg.regime_abnormal_return_z)
    for col in ("regime_trend", "regime_vol"):
        df[col] = reg[col].to_numpy()
    add("regime", "regime_abnormal", reg["regime_abnormal"].astype(float))
    for name in ("BULL_TREND", "BEAR_TREND", "SIDEWAYS"):
        add("regime", f"regime_is_{name.lower()}", (reg["regime_trend"] == name).astype(float))
    for name in ("HIGH_VOL", "LOW_VOL"):
        add("regime", f"regime_is_{name.lower()}", (reg["regime_vol"] == name).astype(float))

    # ---- BTC / ETH context (same bar, both closed at the same decision time)
    if cfg.include_context and context:
        for sym, cdf in context.items():
            tag = sym.replace("USDT", "").lower()
            cc = cdf.set_index("open_time")["close"].reindex(df["open_time"].to_numpy())
            cc = pd.Series(cc.to_numpy(), index=df.index)
            add("context", f"ctx_{tag}_ret_12", cc / cc.shift(12) - 1)
            add("context", f"ctx_{tag}_ret_48", cc / cc.shift(48) - 1)
            add("context", f"ctx_{tag}_rsi", ind.rsi(cc, cfg.rsi_period))
            add("context", f"ctx_{tag}_vol_48", ind.realized_vol(cc, 48))

    out = pd.concat([df, pd.DataFrame(feats)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    keep = [g for g in groups if g in cfg.groups]
    return FeatureFrame(out, {g: groups[g] for g in keep})
