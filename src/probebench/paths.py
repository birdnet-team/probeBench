"""Where probeBench keeps its data.

Everything lives under one data root: ``--data`` on the command line, else the
``PROBEBENCH_DATA`` environment variable, else ``./data``.

    <root>/raw/<dataset>/            downloads (archives are extracted to extracted/)
    <root>/tasks/<task>/task.json    what the task is and how it is scored
    <root>/tasks/<task>/manifest.parquet
    <root>/models/                   model files downloaded on first use
    <root>/cache/<model>/<task>.npz  pooled clip embeddings
    <root>/results/<model>.json      per-task results of a probe run
    <root>/scores/<model>/<task>.npz test-split scores (for the PR figure)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def resolve(cls, data: str | os.PathLike | None = None) -> Paths:
        root = data or os.environ.get("PROBEBENCH_DATA") or "data"
        return cls(Path(root).expanduser().resolve())

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def tasks(self) -> Path:
        return self.root / "tasks"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def scores(self) -> Path:
        return self.root / "scores"
