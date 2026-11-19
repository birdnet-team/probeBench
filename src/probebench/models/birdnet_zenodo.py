"""BirdNET models straight from their Zenodo records, downloaded on first use into
``<data>/models/`` (MD5-checked against the record) and run with a light runtime.

    birdnet-v3.0  BirdNET+ V3.0 Global 12K, FP32 ONNX (onnxruntime), 32 kHz, 3 s;
                  embedding = the "embeddings" output (1,280 values).
                  doi:10.5281/zenodo.24362980, Apache-2.0
    birdnet-v2.4  BirdNET V2.4 Global 6K, FP32 TFLite (ai-edge-litert, or
                  TensorFlow's interpreter), 48 kHz, 3 s; embedding = the
                  GLOBAL_AVG_POOL tensor (1,024 values), as BirdNET-Analyzer reads it.
                  doi:10.5281/zenodo.15050749, CC BY-NC 4.0

Both run on the CPU with ``PROBEBENCH_THREADS`` threads (default 4).
"""

from __future__ import annotations

import os
import warnings
import zipfile
from pathlib import Path

import numpy as np

from ..download import Remote, fetch, zenodo
from . import register

V30 = {
    "record": 24362980,
    "file": "BirdNET+_V3.0_Global_12K_FP32.onnx",
    "doi": "10.5281/zenodo.24362980",
    "license": "Apache-2.0",
}
V24 = {
    "record": 15050749,
    "file": "BirdNET_v2.4_tflite.zip",
    "member": "audio-model.tflite",
    "doi": "10.5281/zenodo.15050749",
    "license": "CC BY-NC 4.0",
}
BATCH = 16


def _threads() -> int:
    return int(os.environ.get("PROBEBENCH_THREADS", 4))


def _remote(record: int, name: str) -> Remote:
    (r,) = zenodo(record, keep=(name,))
    return r


def model_file(spec: dict, models_dir: Path) -> Path:
    """The model file of ``spec``, downloaded and MD5-checked on first use."""
    folder = models_dir / f"zenodo_{spec['record']}"
    target = folder / spec.get("member", spec["file"])
    if target.exists():
        return target
    path = fetch(_remote(spec["record"], spec["file"]), folder)
    if "member" in spec:
        with zipfile.ZipFile(path) as z:
            z.extract(spec["member"], folder)
        path.unlink()
    return target


class BirdNETV30:
    name = "birdnet_v3.0"
    sample_rate = 32000
    window_s = 3.0

    def __init__(self, models_dir: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("birdnet-v3.0 needs: pip install 'probebench[v3]'") from e
        path = model_file(V30, models_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = _threads()
        opts.inter_op_num_threads = 1
        self._s = ort.InferenceSession(
            str(path), opts, providers=["CPUExecutionProvider"]
        )
        self._in = self._s.get_inputs()[0].name
        self.provenance = {k: V30[k] for k in ("doi", "file", "license")}

    def embed(self, x: np.ndarray) -> np.ndarray:
        out = [
            self._s.run(
                ["embeddings"], {self._in: np.ascontiguousarray(x[i : i + BATCH])}
            )[0]
            for i in range(0, len(x), BATCH)
        ]
        return np.concatenate(out).astype(np.float32)


def _interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as e:
            raise ImportError(
                "birdnet-v2.4 needs: pip install 'probebench[v24]'"
            ) from e
    with warnings.catch_warnings():  # preserving all tensors is what we need
        warnings.simplefilter("ignore", UserWarning)
        return Interpreter(
            model_path=str(path),
            num_threads=_threads(),
            experimental_preserve_all_tensors=True,  # to read the pooling tensor
        )


class BirdNETV24:
    name = "birdnet_v2.4"
    sample_rate = 48000
    window_s = 3.0

    def __init__(self, models_dir: Path) -> None:
        self._it = _interpreter(model_file(V24, models_dir))
        self._in = self._it.get_input_details()[0]["index"]
        self._emb = next(
            d["index"]
            for d in self._it.get_tensor_details()
            if d["name"] == "model/GLOBAL_AVG_POOL/Mean"
        )
        self._size = 0
        self.provenance = {k: V24[k] for k in ("doi", "file", "license")}

    def embed(self, x: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(x), BATCH):
            b = np.ascontiguousarray(x[i : i + BATCH], np.float32)
            if len(b) != self._size:
                self._it.resize_tensor_input(self._in, list(b.shape))
                self._it.allocate_tensors()
                self._size = len(b)
            self._it.set_tensor(self._in, b)
            self._it.invoke()
            out.append(self._it.get_tensor(self._emb).copy())
        return np.concatenate(out)


@register("birdnet-v3.0")
def make_v30(models_dir: Path) -> BirdNETV30:
    return BirdNETV30(models_dir)


@register("birdnet-v2.4")
def make_v24(models_dir: Path) -> BirdNETV24:
    return BirdNETV24(models_dir)
