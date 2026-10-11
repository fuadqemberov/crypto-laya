"""Download orchestration: universe inputs, then market data for the selected symbols."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pandas as pd

from ..config import DataConfig
from ..utils.io import write_json, write_parquet
from ..utils.logging import get_logger
from . import store, universe
from .archive import ArchiveClient, resolve_range

log = get_logger("ingestion")


def _progress(done: int, total: int, rec: dict) -> None:
    if done == total or done % 50 == 0:
        log.info("downloaded %d/%d files (last: %s %s)", done, total, rec.get("key"), rec.get("status"))


def kind_timeframes(cfg: DataConfig) -> list[tuple[str, str | None]]:
    """(kind, timeframe) pairs to download for each symbol, as enabled in data.yaml."""
    d = cfg.datasets
    pairs: list[tuple[str, str | None]] = []
    if d.klines:
        pairs += [("klines", tf) for tf in cfg.timeframes]
    for flag, kind in ((d.mark_price_klines, "mark_price_klines"), (d.index_price_klines, "index_price_klines"),
                       (d.premium_index_klines, "premium_index_klines")):
        if flag:
            pairs.append((kind, cfg.base_timeframe))
    for flag, kind in ((d.funding_rate, "funding_rate"), (d.metrics, "metrics"), (d.agg_trades, "agg_trades"),
                       (d.book_depth, "book_depth")):
        if flag:
            pairs.append((kind, None))
    return pairs


async def _download(cfg: DataConfig, jobs: list[tuple[str, str, str | None]], start: date, end: date) -> list[dict]:
    async with ArchiveClient(cfg) as client:
        plans = await asyncio.gather(*(client.plan(k, s, start, end, tf) for k, s, tf in jobs),
                                     return_exceptions=True)
        files = []
        for (k, s, tf), p in zip(jobs, plans, strict=True):
            if isinstance(p, Exception):
                log.error("listing failed for %s %s %s: %r", k, s, tf, p)
                continue
            if not p:
                log.info("no archive files for %s %s %s in range", k, s, tf or "")
            done = store.incorporated_sources(cfg, k, s, tf)
            files += [f for f in p if f.key.rsplit("/", 1)[-1] not in done]  # already in the table
        log.info("planned %d files for %d (kind, symbol, timeframe) jobs", len(files), len(jobs))
        return await client.fetch_many(files, progress=_progress)


def _build(cfg: DataConfig, jobs: list[tuple[str, str, str | None]]) -> list[dict]:
    recs = []
    for k, s, tf in jobs:
        rec = store.build_processed(cfg, k, s, tf)
        if rec:
            recs.append(rec)
    return recs


def build_universe(cfg: DataConfig, max_candidates: int | None = None) -> pd.DataFrame:
    """List every USDT perpetual (delisted included), fetch 1d klines, rank point-in-time."""
    start, end = resolve_range(cfg)
    lookback = timedelta(days=cfg.universe.min_history_days + cfg.universe.rank_window_days + 31)

    async def run():
        async with ArchiveClient(cfg) as c:
            return await c.list_symbols()

    all_symbols = asyncio.run(run())
    cands = universe.candidate_symbols(all_symbols, cfg)
    if max_candidates:
        priority = [s for s in cfg.universe.include_symbols + cfg.context_symbols if s in cands]
        cands = list(dict.fromkeys(priority + cands))[:max_candidates]
    log.info("%d archive symbols, %d USDT perpetual candidates", len(all_symbols), len(cands))
    jobs = [("klines", s, "1d") for s in cands]
    results = asyncio.run(_download(cfg, jobs, start - lookback, end))
    _build(cfg, jobs)
    panel = universe.daily_volume_panel(cfg, cands)
    uni = universe.build_universe(cfg, panel, start, end)
    write_parquet(universe.universe_path(cfg), uni)
    errors = [r for r in results if r["status"] == "error"]
    write_json(cfg.root_path / "universe" / "universe_report.json", {
        "archive_symbols": len(all_symbols), "candidates": len(cands), "months": int(uni["month"].nunique()) if len(uni) else 0,
        "unique_symbols_selected": int(uni["symbol"].nunique()) if len(uni) else 0, "download_errors": errors[:50],
        "rules": cfg.universe.model_dump(), "start": str(start), "end": str(end)})
    return uni


def selected_symbols(cfg: DataConfig) -> list[str]:
    p = universe.universe_path(cfg)
    if not p.exists():
        raise FileNotFoundError("universe not built yet: run `python -m laya_crypto.cli universe` first")
    uni = pd.read_parquet(p)
    syms = list(dict.fromkeys(cfg.context_symbols + sorted(uni["symbol"].unique())))
    return syms


def download_market_data(cfg: DataConfig, symbols: list[str], start: date | None = None,
                         end: date | None = None) -> dict:
    s0, e0 = resolve_range(cfg)
    start, end = start or s0, end or e0
    # extra history so long indicators (EMA200 on 4h) are warm at `start`
    warm_start = start - timedelta(days=45)
    jobs = [(k, s, tf) for s in symbols for k, tf in kind_timeframes(cfg)]
    results = asyncio.run(_download(cfg, jobs, warm_start, end))
    tables = _build(cfg, jobs)
    summary = {"symbols": len(symbols), "files": len(results),
               "ok": sum(r["status"] == "ok" for r in results),
               "missing": sum(r["status"] == "missing" for r in results),
               "errors": [r for r in results if r["status"] == "error"][:100],
               "tables": len(tables), "start": str(warm_start), "end": str(end)}
    write_json(cfg.root_path / "download_report.json", summary)
    return summary
