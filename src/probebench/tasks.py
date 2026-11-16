"""Task definitions: one folder per task under ``<root>/tasks``.

task.json fields:
    name            short id (folder name)
    kind            "multiclass" (one label per clip), "multilabel" (zero or more) or
                    "detection" (binary)
    metric          headline metric: "accuracy", "macro_f1", "map" or "auc" (all
                    metrics defined for the kind are reported anyway)
    description     one or two sentences: source, what is classified, how the splits
                    are grouped
    time_expansion  factor the recording is slowed down by (10 for ultrasonic audio:
                    the native sample rate is divided by it on load, so 20-120 kHz
                    content lands at 2-12 kHz and a 5 ms call lasts 50 ms)
    source / license / citation   provenance, kept with the task

manifest.parquet columns:
    clip_id   unique string
    path      audio file, relative to the task folder (or absolute)
    start_s   start of the clip in seconds of the ORIGINAL recording
    end_s     end of the clip (NaN or <= start_s: to the end of the file)
    label     string (multiclass), list of strings (multilabel; empty = none of the
              classes) or 0/1 (detection)
    split     "train", "val" or "test", fixed and grouped (by recording, site,
              session or individual) so no group spans two splits
    group     the grouping key the split was made on (for reference)
    time_expansion  optional per-clip override of the task's factor
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("multiclass", "multilabel", "detection")
METRICS = ("accuracy", "macro_f1", "map", "auc")


@dataclass
class Task:
    name: str
    kind: str
    metric: str
    folder: Path
    description: str = ""
    time_expansion: float = 1.0
    source: str = ""
    license: str = ""
    citation: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, folder: str | Path) -> Task:
        folder = Path(folder)
        cfg = json.loads((folder / "task.json").read_text())
        names = set(cls.__dataclass_fields__) - {"folder", "extra"}
        known = {k: cfg.pop(k) for k in list(cfg) if k in names}
        t = cls(folder=folder, extra=cfg, **known)
        if t.kind not in KINDS:
            raise ValueError(f"{folder}: kind {t.kind!r} not in {KINDS}")
        if t.metric not in METRICS:
            raise ValueError(f"{folder}: metric {t.metric!r} not in {METRICS}")
        return t

    def manifest(self) -> pd.DataFrame:
        m = pd.read_parquet(self.folder / "manifest.parquet")
        need = {"clip_id", "path", "start_s", "end_s", "label", "split"}
        missing = need - set(m.columns)
        if missing:
            raise ValueError(f"{self.name}: manifest lacks {sorted(missing)}")
        if m.clip_id.duplicated().any():
            raise ValueError(f"{self.name}: duplicate clip_id")
        bad = set(m.split) - {"train", "val", "test"}
        if bad:
            raise ValueError(f"{self.name}: unknown splits {bad}")
        m["path"] = [
            p if Path(p).is_absolute() else str((self.folder / p).resolve())
            for p in m.path
        ]
        return m.reset_index(drop=True)

    def targets(self, m: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Class index per clip (multiclass), a multi-hot matrix (multilabel) or 0/1
        (detection), plus the class names."""
        if self.kind == "multiclass":
            classes = sorted(m.label.astype(str).unique())
            idx = {c: i for i, c in enumerate(classes)}
            return np.array([idx[str(x)] for x in m.label]), classes
        if self.kind == "multilabel":
            classes = sorted({str(c) for x in m.label for c in x})
            idx = {c: i for i, c in enumerate(classes)}
            y = np.zeros((len(m), len(classes)), np.int8)
            for r, x in enumerate(m.label):
                for c in x:
                    y[r, idx[str(c)]] = 1
            return y, classes
        return m.label.astype(int).to_numpy(), ["absent", "present"]


def discover(root: str | Path, names: list[str] | None = None) -> list[Task]:
    """All prepared tasks under ``root`` (a ``tasks`` folder), optionally a subset."""
    root = Path(root)
    tasks = [Task.load(p.parent) for p in sorted(root.glob("*/task.json"))]
    if names:
        by = {t.name: t for t in tasks}
        unknown = [n for n in names if n not in by]
        if unknown:
            raise ValueError(f"tasks not prepared: {unknown}; have {sorted(by)}")
        tasks = [by[n] for n in names]
    return tasks
