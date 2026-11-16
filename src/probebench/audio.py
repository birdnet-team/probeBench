"""Clip loading: mono float32 at the model's sample rate, time expansion applied on
load, and cutting into model-sized windows."""

from __future__ import annotations

import math

import numpy as np
import soundfile as sf
import soxr


def load_clip(
    path: str,
    start_s: float,
    end_s: float,
    sample_rate: int,
    time_expansion: float = 1.0,
) -> np.ndarray:
    """Read [start_s, end_s) of ``path`` (seconds of the original recording), slow it
    down by ``time_expansion`` (the native rate is read as native / factor) and
    resample to ``sample_rate``."""
    info = sf.info(path)
    sr = info.samplerate
    a = int(max(0.0, start_s) * sr)
    to_end = end_s is None or not math.isfinite(end_s) or end_s <= start_s
    b = info.frames if to_end else min(info.frames, int(end_s * sr))
    x, _ = sf.read(path, start=a, stop=max(a + 1, b), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    eff = sr / float(time_expansion)
    if abs(eff - sample_rate) > 1e-6:
        x = soxr.resample(x, eff, sample_rate, quality="HQ").astype(np.float32)
    return x


def windows(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    """[k, n] windows of ``x`` with ``hop``. A clip shorter than one window is
    centered and zero-padded; the last window is aligned to the clip's end so no
    audio is dropped."""
    if len(x) <= n:
        out = np.zeros(n, np.float32)
        o = (n - len(x)) // 2
        out[o : o + len(x)] = x
        return out[None]
    starts = list(range(0, len(x) - n + 1, hop))
    if starts[-1] + n < len(x):
        starts.append(len(x) - n)
    return np.stack([x[s : s + n] for s in starts])
