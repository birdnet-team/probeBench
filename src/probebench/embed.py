"""Embedding a task's clips with a model adapter, cached per (model, task).

Each clip is cut into windows of the model's input length (hop = half a window),
every window is embedded, and the window embeddings are pooled per clip: mean and
max, concatenated, so the probe sees both a summary and the strongest window.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .audio import load_clip, windows
from .models import Adapter
from .tasks import Task

BATCH = 64
# Stop before the machine runs out of memory (useful next to a training run).
MIN_FREE_GB = float(os.environ.get("PROBEBENCH_MIN_FREE_GB", 2))


def available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("inf")


def pool(e: np.ndarray) -> np.ndarray:
    """[mean, max] over a clip's window embeddings. Windows with non-finite values
    are left out (some models return NaN for an all-zero, padded window); a clip
    without any finite window pools to zeros."""
    e = e[np.isfinite(e).all(1)]
    if not len(e):
        return np.zeros(2 * e.shape[1], np.float32)
    return np.concatenate([e.mean(0), e.max(0)])


def embed_task(
    adapter: Adapter,
    task: Task,
    cache_root: str | Path,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """[clips, 2 * D] pooled embeddings aligned with ``task.manifest()``, cached as
    ``<cache_root>/<model>/<task>.npz`` and reused while clips and input format match.
    """
    m = task.manifest()
    out = Path(cache_root) / adapter.name / f"{task.name}.npz"
    te = m["time_expansion"].astype(float).tolist() if "time_expansion" in m else None
    key = {
        "clips": m.clip_id.tolist(),
        "time_expansion": te or task.time_expansion,
        "sample_rate": adapter.sample_rate,
        "window_s": adapter.window_s,
    }
    if out.exists():
        z = np.load(out, allow_pickle=False)
        if json.loads(str(z["key"])) == key:
            return z["emb"]
        log(f"{task.name}: cache {out.name} is stale, embedding again")
    n = round(adapter.window_s * adapter.sample_rate)
    hop = max(1, n // 2)

    def prep(r) -> np.ndarray:
        factor = float(r.time_expansion) if te is not None else task.time_expansion
        x = load_clip(
            r.path, float(r.start_s), float(r.end_s), adapter.sample_rate, factor
        )
        return windows(x, n, hop)

    t0 = time.time()
    pooled: list[np.ndarray | None] = [None] * len(m)
    rows = list(m.itertuples())
    buf: list[np.ndarray] = []
    owners: list[tuple[int, int]] = []

    def flush() -> None:
        e = adapter.embed(np.concatenate(buf))
        o = 0
        for i, k in owners:
            pooled[i] = pool(e[o : o + k])
            o += k
        buf.clear()
        owners.clear()

    with ThreadPoolExecutor(workers) as ex:
        # Bounded read-ahead: a slow model must not let decoded audio pile up.
        ahead = max(4, workers * 4)
        pending = deque(ex.submit(prep, r) for r in rows[:ahead])
        nxt = ahead
        for i in range(len(rows)):
            w = pending.popleft().result()
            if nxt < len(rows):
                pending.append(ex.submit(prep, rows[nxt]))
                nxt += 1
            buf.append(w)
            owners.append((i, len(w)))
            if sum(len(b) for b in buf) >= BATCH:
                flush()
            if i % 100 == 0 and available_gb() < MIN_FREE_GB:
                raise MemoryError(
                    f"{task.name}: less than {MIN_FREE_GB} GB of memory available "
                    "(PROBEBENCH_MIN_FREE_GB)"
                )
            if (i + 1) % 2000 == 0:
                log(f"{task.name}: {i + 1}/{len(m)} clips, {time.time() - t0:.0f} s")
        if buf:
            flush()
    emb = np.stack(pooled).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=emb, key=json.dumps(key))
    log(
        f"{task.name}: embedded {len(m)} clips -> {emb.shape} in {time.time() - t0:.0f} s"
    )
    return emb
