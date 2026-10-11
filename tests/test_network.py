"""Integration with the real public Binance archive (opt-in: `pytest -m network`)."""
from __future__ import annotations

import asyncio
from datetime import date

import pytest

from laya_crypto.config import DataConfig
from laya_crypto.ingestion import store
from laya_crypto.ingestion.archive import ArchiveClient
from laya_crypto.validation.quality import kline_quality

pytestmark = pytest.mark.network


def test_download_verify_parse_real_sample(tmp_path):
    cfg = DataConfig(root=str(tmp_path / "data"))

    async def run():
        async with ArchiveClient(cfg) as c:
            symbols = await c.list_symbols()
            files = await c.plan("klines", "BTCUSDT", date(2024, 1, 1), date(2024, 2, 29), "1d")
            res = await c.fetch_many(files)
            again = await c.fetch_many(files)  # resume: already verified files are skipped
            return symbols, files, res, again

    symbols, files, res, again = asyncio.run(run())
    assert "BTCUSDT" in symbols and len(symbols) > 500
    assert [f.period for f in files] == ["2024-01", "2024-02"]
    assert all(r["status"] == "ok" and r["verified"] for r in res)
    assert {r["sha256"] for r in res} == {r["sha256"] for r in again}
    rec = store.build_processed(cfg, "klines", "BTCUSDT", "1d")
    assert rec["rows"] == 60 and rec["duplicates_dropped"] == 0
    q = kline_quality(store.load(cfg, "klines", "BTCUSDT", "1d"), "1d", "BTCUSDT")
    assert q["missing_bars"] == 0 and q["ohlc_violations"] == 0
