"""Fetch an official LAYA base checkpoint into models/base/<name>, pinned and hash-verified.

`laya.load("english")` resolves through the Hugging Face API, which rate-limits shared IPs. This
module downloads the same files at the revision LAYA itself pins (`laya.PINNED_REVISIONS`), checks
each LFS file against the SHA-256 the Hub publishes, and writes a manifest. Training and
inference then point at the local directory, so no run silently picks up a different revision.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

from ..utils.io import write_json
from ..utils.logging import get_logger

log = get_logger("base_checkpoint")

HUB = "https://huggingface.co"
# Built-in names -> (repo, subfolder), mirroring laya.router.DEFAULT_MODELS.
BUILTIN = {
    "english": ("convaiinnovations/laya", ""),
    "multilingual": ("convaiinnovations/laya", "multilingual"),
    "typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}
NEEDED_TOP = ("rl_agent_config.json", "model.safetensors")
NEEDED_DIRS = ("tokenizer/", "encoder/")


def pinned_revision(repo: str) -> str:
    from laya.revisions import PINNED_REVISIONS

    return PINNED_REVISIONS.get(repo, "main")


def _get(client: httpx.Client, url: str, retries: int = 6) -> httpx.Response:
    delay = 2.0
    for attempt in range(retries):
        try:
            r = client.get(url, follow_redirects=True)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == retries - 1:
                raise
            log.warning("GET %s failed (%s); retry in %.0fs", url, e, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def list_files(client: httpx.Client, repo: str, revision: str, subfolder: str) -> list[dict]:
    url = f"{HUB}/api/models/{repo}/tree/{revision}"
    if subfolder:
        url += f"/{subfolder}"
    files = _get(client, url + "?recursive=true").json()
    prefix = (subfolder + "/") if subfolder else ""
    out = []
    for f in files:
        if f.get("type") != "file":
            continue
        rel = f["path"][len(prefix):] if f["path"].startswith(prefix) else f["path"]
        if rel in NEEDED_TOP or any(rel.startswith(d) and rel.count("/") == 1 for d in NEEDED_DIRS):
            out.append({"path": f["path"], "rel": rel, "size": f.get("size"),
                        "sha256": (f.get("lfs") or {}).get("oid")})
    return out


def _download(client: httpx.Client, url: str, dest: Path, size: int | None, sha256: str | None) -> None:
    if dest.exists() and (size is None or dest.stat().st_size == size):
        if sha256 is None or _sha(dest) == sha256:
            return
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with client.stream("GET", url, headers=headers, follow_redirects=True) as r:
                if r.status_code == 416:
                    break
                if r.status_code not in (200, 206):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                mode = "ab" if r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            log.warning("download %s interrupted (%s); resuming", url, e)
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"could not download {url}")
    if size is not None and part.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: size {part.stat().st_size} != expected {size}")
    if sha256 is not None and _sha(part) != sha256:
        part.unlink()
        raise RuntimeError(f"{dest.name}: SHA-256 mismatch, partial file removed")
    part.replace(dest)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def fetch_base_checkpoint(name: str = "english", dest_root: str | Path = "models/base",
                          token: str | None = None) -> Path:
    """Download (or verify) a base checkpoint and return its local directory."""
    if name not in BUILTIN:
        raise ValueError(f"unknown base checkpoint {name!r}; choose one of {sorted(BUILTIN)}")
    repo, sub = BUILTIN[name]
    rev = pinned_revision(repo)
    dest = Path(dest_root) / name
    manifest_path = dest / "laya_base_manifest.json"
    if manifest_path.exists() and (dest / "model.safetensors").exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("revision") == rev:
            return dest
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=httpx.Timeout(60, read=300), headers=headers) as client:
        files = list_files(client, repo, rev, sub)
        missing = [n for n in NEEDED_TOP if n not in {f["rel"] for f in files}]
        if missing:
            raise RuntimeError(f"{repo}@{rev}/{sub}: missing {missing}")
        for f in files:
            log.info("fetching %s (%s bytes)", f["path"], f["size"])
            _download(client, f"{HUB}/{repo}/resolve/{rev}/{f['path']}", dest / f["rel"], f["size"], f["sha256"])
    write_json(manifest_path, {"name": name, "repo": repo, "subfolder": sub, "revision": rev,
                               "files": files, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return dest


def resolve_base(base: str, dest_root: str | Path = "models/base") -> str:
    """A local directory for `base`: an existing checkpoint dir, or a fetched built-in name."""
    p = Path(base)
    if p.is_dir() and (p / "rl_agent_config.json").exists():
        return str(p)
    return str(fetch_base_checkpoint(base, dest_root))
