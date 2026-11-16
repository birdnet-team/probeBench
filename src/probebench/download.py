"""Resumable downloads with size/MD5 checks, Zenodo record listing, and archive
extraction that verifies every member before the archive is removed."""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

CHUNK = 1 << 20
RETRIES = 5


@dataclass(frozen=True)
class Remote:
    """One file of a dataset. ``extract``: unpack into ``extracted/`` after download."""

    name: str
    url: str
    size: int | None = None
    md5: str | None = None
    extract: bool = False


def zenodo(
    record: int, keep: tuple[str, ...] = (), drop: tuple[str, ...] = ()
) -> list[Remote]:
    """The files of a public Zenodo record (``keep``: only names starting with one of
    these; ``drop``: skip names starting with one of these)."""
    r = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
    r.raise_for_status()
    out = []
    for f in r.json()["files"]:
        k = f["key"]
        if keep and not k.startswith(keep):
            continue
        if drop and k.startswith(drop):
            continue
        algo, _, digest = f["checksum"].partition(":")
        out.append(
            Remote(
                k,
                f["links"]["self"],
                int(f["size"]),
                digest if algo == "md5" else None,
                extract=k.endswith((".zip", ".tar.gz", ".tgz")),
            )
        )
    return out


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _get(url: str, path: Path, size: int | None) -> None:
    have = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if have and r.status_code == 200:  # no range support: start over
            have = 0
        elif r.status_code == 416:  # already complete
            return
        r.raise_for_status()
        total = size or (have + int(r.headers.get("content-length", 0)) or None)
        with (
            open(path, "ab" if have else "wb") as f,
            tqdm(
                total=total, initial=have, unit="B", unit_scale=True, desc=path.name
            ) as bar,
        ):
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                bar.update(len(chunk))


def fetch(remote: Remote, folder: Path, log: Callable[[str], None] = print) -> Path:
    """Download ``remote`` into ``folder`` (resuming a partial file) and verify it."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / remote.name
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        try:
            _get(remote.url, path, remote.size)
        except requests.RequestException as e:
            log(f"{remote.name}: {e} (attempt {attempt}/{RETRIES})")
            time.sleep(min(60, 5 * attempt))
            continue
        if remote.size is not None and path.stat().st_size != remote.size:
            log(f"{remote.name}: size {path.stat().st_size} != {remote.size}, retrying")
            if path.stat().st_size > remote.size:
                path.unlink()
            continue
        if remote.md5 and md5sum(path) != remote.md5:
            log(f"{remote.name}: MD5 mismatch, downloading again")
            path.unlink()
            continue
        return path
    raise RuntimeError(f"could not download {remote.url}")


def _members(archive: Path) -> list[tuple[str, int]]:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            return [(i.filename, i.file_size) for i in z.infolist() if not i.is_dir()]
    with tarfile.open(archive) as t:
        return [(m.name, m.size) for m in t.getmembers() if m.isfile()]


def extract(
    archive: Path,
    dest: Path,
    keep_archive: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Unpack ``archive`` into ``dest``, check every member (path and size), then
    delete the archive unless ``keep_archive``."""
    dest.mkdir(parents=True, exist_ok=True)
    log(f"extracting {archive.name}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    mem = _members(archive)
    bad = [
        m for m, s in mem if not (dest / m).exists() or (dest / m).stat().st_size != s
    ]
    if bad:
        raise RuntimeError(
            f"{archive.name}: {len(bad)} members missing after extraction"
        )
    if not keep_archive:
        archive.unlink()


def fetch_all(
    remotes: list[Remote],
    folder: Path,
    keep_archives: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Download (and extract) every remote once; ``.done`` markers make reruns cheap."""
    for r in remotes:
        done = folder / f".{r.name}.done"
        if done.exists():
            continue
        path = fetch(r, folder, log)
        if r.extract:
            extract(path, folder / "extracted", keep_archive=keep_archives, log=log)
        done.write_text(json.dumps({"url": r.url, "size": r.size, "md5": r.md5}))
