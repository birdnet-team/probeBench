"""Perch 2.0 (Google DeepMind), the official TensorFlow SavedModel from Kaggle Models
(google/bird-vocalization-classifier, TensorFlow2 variants perch_v2 for GPU and
perch_v2_cpu), downloaded anonymously on first use into ``<data>/models/`` with the
archive's size and MD5 checked. 32 kHz, 5 s windows; embedding = the "embedding"
output (1,536 values). Apache-2.0.

The GPU variant is used when TensorFlow sees a GPU, the CPU variant otherwise
(``PROBEBENCH_PERCH_VARIANT=perch_v2|perch_v2_cpu`` overrides).
``PROBEBENCH_TF_GPU_MB`` caps TensorFlow's GPU memory.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import numpy as np

from ..download import Remote, fetch
from . import register

KAGGLE = "https://www.kaggle.com/api/v1/models/google/bird-vocalization-classifier/tensorFlow2"
VARIANTS = {
    "perch_v2": {
        "version": 2,
        "size": 379632720,
        "md5": "0c08776fbc30640984dcc2fe8447d236",
    },
    "perch_v2_cpu": {
        "version": 1,
        "size": 379635836,
        "md5": "ded815f71094ad4493ff41a870b27b90",
    },
}
BATCH = 16


def _tf():
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("perch-v2 needs: pip install 'probebench[perch]'") from e
    mb = int(os.environ.get("PROBEBENCH_TF_GPU_MB", "0"))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and mb:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=mb)]
        )
    return tf, bool(gpus)


def saved_model_dir(variant: str, models_dir: Path) -> Path:
    """The extracted SavedModel of ``variant``, downloaded on first use."""
    v = VARIANTS[variant]
    folder = models_dir / f"kaggle_{variant}_v{v['version']}"
    found = sorted(folder.rglob("saved_model.pb"))
    if found:
        return found[0].parent
    name = f"{variant}_v{v['version']}.tar.gz"
    url = f"{KAGGLE}/{variant}/{v['version']}/download"
    archive = fetch(Remote(name, url, v["size"], v["md5"]), folder)
    with tarfile.open(archive) as t:
        t.extractall(folder, filter="data")
    archive.unlink()
    return sorted(folder.rglob("saved_model.pb"))[0].parent


class PerchV2:
    name = "perch_v2"
    sample_rate = 32000
    window_s = 5.0

    def __init__(self, models_dir: Path) -> None:
        self._tf, gpu = _tf()
        variant = os.environ.get("PROBEBENCH_PERCH_VARIANT") or (
            "perch_v2" if gpu else "perch_v2_cpu"
        )
        model = self._tf.saved_model.load(str(saved_model_dir(variant, models_dir)))
        self._fn = model.signatures["serving_default"]
        self._model = model  # keeps the signature's variables alive
        self.provenance = {
            "source": f"Kaggle google/bird-vocalization-classifier/tensorFlow2/{variant}",
            "version": VARIANTS[variant]["version"],
            "md5": VARIANTS[variant]["md5"],
            "license": "Apache-2.0",
        }

    def embed(self, x: np.ndarray) -> np.ndarray:
        tf = self._tf
        out = []
        for i in range(0, len(x), BATCH):
            b = tf.constant(np.ascontiguousarray(x[i : i + BATCH], np.float32))
            out.append(self._fn(inputs=b)["embedding"].numpy())
        return np.concatenate(out).astype(np.float32)


@register("perch-v2")
def make(models_dir: Path) -> PerchV2:
    return PerchV2(models_dir)
