"""TorchScript models that take raw audio [B, samples] and return (scores,
embeddings), such as the BirdNET+ V3.0 release files (32 kHz, 3 s). The embedding
is the second output."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class TorchScriptAdapter:
    def __init__(
        self,
        path: str,
        name: str | None = None,
        sample_rate: int = 32000,
        window_s: float = 3.0,
        batch: int = 64,
    ) -> None:
        import torch

        self._torch = torch
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        frac = float(os.environ.get("PROBEBENCH_GPU_FRACTION", "0"))
        if frac and self._dev == "cuda":
            torch.cuda.set_per_process_memory_fraction(frac)
        self._m = torch.jit.load(path, map_location=self._dev).eval()
        self._batch = batch
        self.sample_rate = sample_rate
        self.window_s = window_s
        self.name = name or Path(path).stem

    def embed(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self._batch):
                b = np.ascontiguousarray(x[i : i + self._batch], np.float32)
                _, e = self._m(torch.from_numpy(b).to(self._dev))
                out.append(e.float().cpu().numpy())
        return np.concatenate(out)
