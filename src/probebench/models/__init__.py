"""Model adapters: everything probeBench needs to know about a model.

An adapter is any object with

    name: str          unique id, used for the cache and results file names
    sample_rate: int   input sample rate in Hz
    window_s: float    input window length in seconds
    embed(x) -> np.ndarray   float32 [N, window samples] -> float32 [N, D]

Built-in specs for ``probebench run --model``:

    mel                      log-mel statistics, no learning (the floor)
    birdnet-v3.0             BirdNET V3.0 through the birdnet package (ONNX)
    birdnet-v2.4             BirdNET V2.4 through the birdnet package (TensorFlow)
    torchscript:<file.pt>    a TorchScript model taking raw audio and returning
                             (scores, embeddings), e.g. a BirdNET+ V3.0 release file
    py:<module>:<factory>    your own: ``factory()`` returns an adapter

Adding a model is one small class: see ``mel.py`` for a complete example, and
``register`` to give it a short name.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Adapter(Protocol):
    name: str
    sample_rate: int
    window_s: float

    def embed(self, x: np.ndarray) -> np.ndarray: ...


_REGISTRY: dict[str, Callable[[], Adapter]] = {}


def register(spec: str) -> Callable:
    """Decorator: ``@register("my-model")`` makes ``--model my-model`` build it."""

    def wrap(factory: Callable[[], Adapter]) -> Callable[[], Adapter]:
        _REGISTRY[spec] = factory
        return factory

    return wrap


def available() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)


def _load_builtins() -> None:
    from . import birdnet_pkg, mel  # noqa: F401  (registration side effect)


def from_spec(spec: str) -> Adapter:
    """Build the adapter a ``--model`` spec names (downloads weights on first use)."""
    _load_builtins()
    if spec in _REGISTRY:
        return _REGISTRY[spec]()
    if spec.startswith("torchscript:"):
        from .torchscript import TorchScriptAdapter

        path, _, name = spec[len("torchscript:") :].partition("@")
        return TorchScriptAdapter(path, name or None)
    if spec.startswith("py:"):
        _, mod, fac = spec.split(":", 2)
        return getattr(importlib.import_module(mod), fac)()
    raise ValueError(f"unknown model {spec!r}; built-in: {available()}")
