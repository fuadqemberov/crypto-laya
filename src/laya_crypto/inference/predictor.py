"""Local inference: latest closed candle -> features -> calibrated LONG/SHORT/WAIT.

Data comes from the local archive cache, topped up with the most recent closed candles from the
public Binance REST API. If the live API cannot be reached the result is computed on cached data
and explicitly marked `stale`; if even that is too old or missing, an error is returned. No
prediction in this path is ever hard-coded.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config, load_features_config, load_inference_config, project_path
from ..datasets.serialize import STATE_FIELDS, serialize_row
from ..features.pipeline import AuxData, build_features
from ..ingestion import live, store
from ..labeling.triple_barrier import CLASSES
from ..utils.timeframes import tf_ms

CORE_FEATURES = ("rsi", "atr_pct", "ema21_dist_atr", "adx", "bb_width", "rel_volume", "htf_1h_trend")


class InferenceError(RuntimeError):
    pass


def _merge(cached: pd.DataFrame | None, fresh: pd.DataFrame | None, key: str) -> pd.DataFrame | None:
    frames = [f for f in (cached, fresh) if f is not None and len(f)]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=[key], keep="last").sort_values(key).reset_index(drop=True)


class Predictor:
    def __init__(self, checkpoint: str | None = None, device: str | None = None):
        self.icfg = load_inference_config()
        self.dcfg = load_data_config()
        self.fcfg = load_features_config()
        ck = Path(checkpoint or self.icfg.checkpoint)
        self.checkpoint = ck if ck.is_absolute() else project_path(str(ck))
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from .laya_model import LayaDecisionModel

            if not (self.checkpoint / "rl_agent_config.json").exists():
                raise InferenceError(f"no trained checkpoint at {self.checkpoint}. Train a profile and run "
                                     "`python -m laya_crypto.cli select-best`, or pass --checkpoint.")
            self._model = LayaDecisionModel(self.checkpoint, self.device)
        return self._model

    # ----------------------------------------------------------------- data
    def market_data(self, symbol: str, tf: str, offline: bool
                    ) -> tuple[pd.DataFrame, AuxData, dict[str, pd.DataFrame], dict, list[str]]:
        notes: list[str] = []
        cfg = self.dcfg
        base = store.load(cfg, "klines", symbol, tf)
        funding, metrics = store.load(cfg, "funding_rate", symbol), store.load(cfg, "metrics", symbol)
        mark = store.load(cfg, "mark_price_klines", symbol, tf)
        index = store.load(cfg, "index_price_klines", symbol, tf)
        premium = store.load(cfg, "premium_index_klines", symbol, tf)
        source = "archive"
        if not offline:
            try:
                base = _merge(base, live.fetch_recent_klines(cfg, symbol, tf, 1500), "open_time")
                funding = _merge(funding, live.fetch_funding_history(cfg, symbol), "funding_time")
                metrics = _merge(metrics, live.fetch_open_interest_hist(cfg, symbol, tf if tf in ("5m", "15m", "30m", "1h")
                                                                         else "5m"), "time")
                for kind in ("mark_price_klines", "index_price_klines", "premium_index_klines"):
                    fresh = live.fetch_recent_klines(cfg, symbol, tf, 1500, kind)
                    if kind == "mark_price_klines":
                        mark = _merge(mark, fresh, "open_time")
                    elif kind == "index_price_klines":
                        index = _merge(index, fresh, "open_time")
                    else:
                        premium = _merge(premium, fresh, "open_time")
                source = "archive+live"
            except live.LiveDataError as e:
                notes.append(f"live data unavailable: {e}")
        if base is None or len(base) < self.icfg.warmup_bars // 4:
            raise InferenceError(f"{symbol}: no cached {tf} candles and no live data. Run `download --symbols {symbol}`.")
        base = base.tail(self.icfg.warmup_bars + 50)
        context = {}
        for s in self.dcfg.context_symbols:
            c = base if s == symbol else store.load(cfg, "klines", s, tf)
            if s != symbol and not offline and source == "archive+live":
                try:
                    c = _merge(c, live.fetch_recent_klines(cfg, s, tf, 1500), "open_time")
                except live.LiveDataError as e:
                    notes.append(f"context {s} live data unavailable: {e}")
            if c is not None:
                context[s] = c[["open_time", "close"]]
        aux = AuxData(funding=funding, metrics=metrics, mark=mark, index=index, premium=premium,
                      metrics_lag_ms=cfg.metrics_publication_lag_ms)
        return base, aux, context, {"source": source}, notes

    # ----------------------------------------------------------------- predict
    def features(self, symbol: str, tf: str, offline: bool):
        t0 = time.perf_counter()
        base, aux, context, meta, notes = self.market_data(symbol, tf, offline)
        t1 = time.perf_counter()
        ff = build_features(base, self.fcfg, tf, aux, context)
        row = ff.frame.iloc[-1]
        t2 = time.perf_counter()
        return row, meta, notes, (t1 - t0) * 1000, (t2 - t1) * 1000

    def predict_one(self, symbol: str, tf: str | None = None, offline: bool = False) -> dict:
        tf = tf or self.icfg.timeframe
        t_start = time.perf_counter()
        try:
            row, meta, notes, data_ms, feat_ms = self.features(symbol, tf, offline)
        except InferenceError as e:
            return {"symbol": symbol, "timeframe": tf, "error": str(e)}
        if not bool(row.get("bar_present", True)):
            return {"symbol": symbol, "timeframe": tf, "error": "latest candle missing in data"}
        close_ms = int(row["open_time"]) + tf_ms(tf)
        age_min = (time.time() * 1000 - close_ms) / 60_000
        stale = age_min > self.icfg.max_data_age_minutes
        missing = [c for c in CORE_FEATURES if c not in row or not np.isfinite(float(row[c]))]
        all_fields = [col for _, col, _ in STATE_FIELDS]
        na_fields = [c for c in all_fields if c not in row or pd.isna(row[c])]
        if missing:
            return {"symbol": symbol, "timeframe": tf, "error": f"core features missing (warm-up/data gaps): {missing}",
                    "data_timestamp_utc": str(pd.to_datetime(close_ms, unit="ms", utc=True))}
        state = serialize_row(row, symbol, tf)
        t_m = time.perf_counter()
        proba = self.model.predict_proba([state], batch_size=1)[0]
        model_ms = (time.perf_counter() - t_m) * 1000
        probs = {c: round(float(p), 4) for c, p in zip(CLASSES, proba, strict=True)}
        top = max(probs, key=probs.get)
        abstained = probs[top] < self.icfg.abstain_below
        return {
            "symbol": symbol, "timeframe": tf, "decision": "WAIT" if abstained else top,
            "model_argmax": top, "abstained": abstained, "abstain_below": self.icfg.abstain_below,
            "probabilities": probs, "model_version": self.model.version,
            "data_timestamp_utc": str(pd.to_datetime(close_ms, unit="ms", utc=True)),
            "data_age_minutes": round(age_min, 1), "stale": stale, "data_source": meta["source"],
            "missing_state_fields": na_fields, "notes": notes,
            "latency_ms": {"data": round(data_ms, 1), "features": round(feat_ms, 1), "model": round(model_ms, 1),
                           "total": round((time.perf_counter() - t_start) * 1000, 1)},
            "disclaimer": "research output; probabilities are calibrated classifier scores, not win probabilities",
        }

    def predict_symbols(self, symbols: list[str], tf: str | None = None, offline: bool = False) -> list[dict]:
        return [self.predict_one(s, tf, offline) for s in symbols]

    def benchmark(self, symbols: list[str], n: int = 20, offline: bool = True) -> dict:
        sym = symbols[0]
        tf = self.icfg.timeframe
        row, *_ = self.features(sym, tf, offline)
        state = serialize_row(row, sym, tf)
        self.model.predict_proba([state], 1)  # warm-up
        single = []
        for _ in range(n):
            t = time.perf_counter()
            self.model.predict_proba([state], 1)
            single.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        self.model.predict_proba([state] * 32, 32)
        batch32 = (time.perf_counter() - t) * 1000
        e2e = []
        for _ in range(max(3, n // 5)):
            t = time.perf_counter()
            self.predict_one(sym, tf, offline)
            e2e.append((time.perf_counter() - t) * 1000)
        return {"symbol": sym, "device": str(self.model.agent.device) if hasattr(self.model.agent, "device") else self.device,
                "model_only_ms": {"p50": float(np.median(single)), "p95": float(np.percentile(single, 95)), "n": n},
                "batch32_ms_per_state": batch32 / 32,
                "end_to_end_ms": {"p50": float(np.median(e2e)), "max": float(max(e2e)), "n": len(e2e),
                                  "offline": offline},
                "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
