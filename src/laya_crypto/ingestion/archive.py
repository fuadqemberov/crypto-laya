"""Binance public archive (data.binance.vision) client for USD-M futures.

- Availability comes from the bucket listing, so delisted contracts are visible and no request is
  made for a file that does not exist.
- Every zip is verified against the `.CHECKSUM` file Binance publishes next to it.
- Downloads are concurrent but bounded, retried with exponential backoff, written to `.part`
  files and renamed only after verification, so an interrupted run resumes cleanly.
"""
from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from ..config import DataConfig
from ..utils.logging import get_logger

log = get_logger("archive")

# kind -> (granularity, archive folder, has timeframe subfolder)
KINDS: dict[str, tuple[str, str, bool]] = {
    "klines": ("monthly", "klines", True),
    "mark_price_klines": ("monthly", "markPriceKlines", True),
    "index_price_klines": ("monthly", "indexPriceKlines", True),
    "premium_index_klines": ("monthly", "premiumIndexKlines", True),
    "funding_rate": ("monthly", "fundingRate", False),
    "agg_trades": ("monthly", "aggTrades", False),
    "metrics": ("daily", "metrics", False),
    "book_depth": ("daily", "bookDepth", False),
}
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_DATE_RE = re.compile(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$")


@dataclass(frozen=True)
class ArchiveFile:
    kind: str
    symbol: str
    timeframe: str | None
    period: str  # YYYY-MM or YYYY-MM-DD
    granularity: str
    key: str  # bucket key of the zip

    def url(self, base: str) -> str:
        return f"{base}/{self.key}"

    def local_path(self, raw_root: Path) -> Path:
        tf = self.timeframe or "_"
        return raw_root / self.kind / tf / self.symbol / Path(self.key).name


def _parse_period(p: str) -> date:
    return date.fromisoformat(p if len(p) == 10 else p + "-01")


class ArchiveClient:
    def __init__(self, cfg: DataConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.raw_root = cfg.raw_path
        self._client = client
        self._sem = asyncio.Semaphore(cfg.download.concurrency)

    async def __aenter__(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.download.timeout_s),
                                             limits=httpx.Limits(max_connections=self.cfg.download.concurrency * 2),
                                             headers={"User-Agent": "laya-crypto-trainer/0.1 (research)"})
        return self

    async def __aexit__(self, *exc):
        if self._client is not None:
            await self._client.aclose()

    # ------------------------------------------------------------------ http with backoff
    async def _get(self, url: str, **kw) -> httpx.Response:
        d = self.cfg.download
        for attempt in range(d.retries + 1):
            try:
                async with self._sem:
                    r = await self._client.get(url, **kw)
                if r.status_code == 404:
                    return r
                if r.status_code in (418, 429) or r.status_code >= 500:
                    retry_after = float(r.headers.get("Retry-After", 0) or 0)
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r) \
                        if not retry_after else _RetryAfter(retry_after)
                r.raise_for_status()
                return r
            except _RetryAfter as ra:
                await asyncio.sleep(ra.seconds)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == d.retries:
                    raise
                delay = d.backoff_base_s * 2 ** attempt * (1 + random.random() * 0.25)
                log.warning("GET %s failed (%s), retry %d/%d in %.1fs", url, e, attempt + 1, d.retries, delay)
                await asyncio.sleep(delay)
        raise RuntimeError(f"GET {url}: retries exhausted (rate limited)")

    # ---------------------------------------------------------------------------- listing
    async def list_prefix(self, prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
        """(common prefixes, keys) under `prefix`, following S3 pagination."""
        prefixes, keys, marker = [], [], None
        while True:
            params = {"prefix": prefix}
            if delimiter:
                params["delimiter"] = delimiter
            if marker:
                params["marker"] = marker
            r = await self._get(self.cfg.listing_base_url, params=params)
            root = ET.fromstring(r.text)
            page_keys = [e.text for e in root.iter(f"{_S3_NS}Key")]
            page_prefixes = [e.text for e in root.iter(f"{_S3_NS}Prefix") if e.text != prefix]
            keys += page_keys
            prefixes += [p for p in page_prefixes if p not in prefixes]
            truncated = (root.findtext(f"{_S3_NS}IsTruncated") or "false") == "true"
            if not truncated:
                return prefixes, keys
            marker = root.findtext(f"{_S3_NS}NextMarker") or (page_keys[-1] if page_keys else
                                                              (page_prefixes[-1] if page_prefixes else None))
            if marker is None:
                return prefixes, keys

    async def list_symbols(self) -> list[str]:
        prefixes, _ = await self.list_prefix("data/futures/um/monthly/klines/", delimiter="/")
        return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes)

    async def available_files(self, kind: str, symbol: str, timeframe: str | None = None,
                              granularity: str | None = None) -> list[ArchiveFile]:
        gran, folder, has_tf = KINDS[kind]
        gran = granularity or gran
        if has_tf and not timeframe:
            raise ValueError(f"{kind} needs a timeframe")
        prefix = f"data/futures/um/{gran}/{folder}/{symbol}/" + (f"{timeframe}/" if has_tf else "")
        _, keys = await self.list_prefix(prefix)
        files = []
        for k in keys:
            if not k.endswith(".zip"):
                continue
            m = _DATE_RE.search(k)
            if m:
                files.append(ArchiveFile(kind, symbol, timeframe if has_tf else None, m.group(1), gran, k))
        return sorted(files, key=lambda f: f.period)

    async def plan(self, kind: str, symbol: str, start: date, end: date,
                   timeframe: str | None = None) -> list[ArchiveFile]:
        """Files covering [start, end]: monthly archives where they exist, daily ones for the rest."""
        gran, _, _ = KINDS[kind]
        if gran == "daily":
            return [f for f in await self.available_files(kind, symbol, timeframe)
                    if start <= _parse_period(f.period) <= end]
        monthly = [f for f in await self.available_files(kind, symbol, timeframe)
                   if date(start.year, start.month, 1) <= _parse_period(f.period) <= end]
        covered_until = None
        if monthly:
            last = _parse_period(monthly[-1].period)
            covered_until = (last.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if covered_until is None or covered_until < end:
            daily_start = (covered_until + timedelta(days=1)) if covered_until else start
            try:
                daily = await self.available_files(kind, symbol, timeframe, granularity="daily")
            except httpx.HTTPError:
                daily = []
            monthly += [f for f in daily if daily_start <= _parse_period(f.period) <= end]
        return monthly

    # --------------------------------------------------------------------------- download
    async def fetch(self, f: ArchiveFile) -> tuple[Path, str] | None:
        """Download and verify one file; returns (path, sha256) or None when it does not exist."""
        dest = f.local_path(self.raw_root)
        sidecar = dest.with_suffix(".zip.sha256")
        if dest.exists() and sidecar.exists():
            return dest, sidecar.read_text().strip()
        expected = None
        if self.cfg.download.verify_checksums:
            cr = await self._get(f.url(self.cfg.archive_base_url) + ".CHECKSUM")
            if cr.status_code == 200:
                expected = cr.text.split()[0].strip().lower()
        r = await self._get(f.url(self.cfg.archive_base_url))
        if r.status_code == 404:
            return None
        data = r.content
        digest = hashlib.sha256(data).hexdigest()
        if expected and digest != expected:
            raise ChecksumError(f"{f.key}: sha256 {digest} != published {expected}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".zip.part")
        part.write_bytes(data)
        part.replace(dest)
        sidecar.write_text(digest + ("" if expected else " unverified") + "\n")
        return dest, digest

    async def fetch_many(self, files: list[ArchiveFile], progress=None) -> list[dict]:
        results: list[dict] = []
        done = 0

        async def one(f: ArchiveFile):
            nonlocal done
            try:
                out = await self.fetch(f)
                rec = {"key": f.key, "kind": f.kind, "symbol": f.symbol, "timeframe": f.timeframe,
                       "period": f.period, "status": "ok" if out else "missing"}
                if out:
                    rec.update(path=str(out[0]), sha256=out[1].split()[0],
                               verified="unverified" not in out[1])
            except Exception as e:  # noqa: BLE001 - recorded per file, the batch continues
                rec = {"key": f.key, "kind": f.kind, "symbol": f.symbol, "status": "error", "error": repr(e)}
                log.error("failed %s: %r", f.key, e)
            results.append(rec)
            done += 1
            if progress:
                progress(done, len(files), rec)

        await asyncio.gather(*(one(f) for f in files))
        return results


class ChecksumError(RuntimeError):
    pass


class _RetryAfter(Exception):
    def __init__(self, seconds: float):
        self.seconds = seconds


def resolve_range(cfg: DataConfig) -> tuple[date, date]:
    start = date.fromisoformat(cfg.start)
    if cfg.end:
        end = date.fromisoformat(cfg.end)
    else:
        end = (datetime.now(UTC) - timedelta(days=1)).date()
    if end < start:
        raise ValueError(f"data.end {end} is before data.start {start}")
    return start, end


def utc_now_ms() -> int:
    return int(time.time() * 1000)
