"""BirdNET models through the ``birdnet`` package, which downloads them on first use
(cache: ``BIRDNET_APP_DATA``).

The embedding is the model's own embedding output (1,280 values for V3.0, 1,024 for
V2.4) for each 3 s window, as BirdNET-Analyzer computes it. The package applies
its default bandpass (0-15 kHz) to every window. The session runs one worker
with small batches: each birdnet worker process holds its own model copy (about
1 GB), and the package's default worker count can exhaust a machine's memory.

The package version decides which build "3.0" means: birdnet 1.4 serves
BirdNET+ V3.0-preview3.1 (11K classes). Results record the package version and
the model file so runs stay comparable.
"""

from __future__ import annotations

import atexit
from importlib import metadata
from pathlib import Path

import numpy as np

from . import register


class BirdNETPackage:
    window_s = 3.0

    def __init__(self, version: str, backend: str, batch: int = 16) -> None:
        try:
            import birdnet
        except ImportError as e:
            extra = "birdnet" if backend == "onnx" else "birdnet-tf"
            raise ImportError(
                f"BirdNET {version} needs the birdnet package: "
                f"pip install 'probebench[{extra}]'"
            ) from e
        self.model = birdnet.load("acoustic", version, backend)
        self.sample_rate = int(self.model.get_sample_rate())
        self.name = f"birdnet_v{version}"
        self._ctx = self.model.encode_session(
            batch_size=batch, n_workers=1, n_producers=1
        )
        self._session = self._ctx.__enter__()
        atexit.register(self.close)
        self.provenance = {
            "birdnet_package": metadata.version("birdnet"),
            "model_file": Path(str(getattr(self.model, "model_path", ""))).name,
        }

    def embed(self, x: np.ndarray) -> np.ndarray:
        r = self._session.run_arrays([(w, self.sample_rate) for w in x])
        if r.embeddings.shape[1] != 1:
            raise ValueError("expected one segment per window")
        return np.asarray(r.embeddings[:, 0, :], np.float32)

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None


@register("birdnet-v3.0")
def make_v30() -> BirdNETPackage:
    return BirdNETPackage("3.0", "onnx")


@register("birdnet-v2.4")
def make_v24() -> BirdNETPackage:
    return BirdNETPackage("2.4", "tf")
