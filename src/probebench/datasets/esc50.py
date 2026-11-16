"""ESC-50 (Piczak 2015): 50 environmental sound classes (animals, nature, human,
domestic, urban), 5 s clips. Clips whose Freesound source recording also appears in
FSD50K or UrbanSound8K (common training data for audio models) are dropped; the list
ships as esc50_excluded.txt. Official folds: 5 test, 4 val, 1-3 train."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pandas as pd

from ..download import Remote
from .common import Dataset, write_task

URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def excluded() -> set[str]:
    text = resources.files(__package__).joinpath("esc50_excluded.txt").read_text()
    return {s for s in text.split() if s.endswith(".wav")}


def prepare(raw: Path, tasks: Path) -> None:
    src = raw / "extracted" / "ESC-50-master"
    a = pd.read_csv(src / "meta" / "esc50.csv")
    a = a[~a.filename.isin(excluded())]
    m = pd.DataFrame(
        {
            "clip_id": a.filename.astype(str),
            "path": [str(src / "audio" / f) for f in a.filename],
            "label": a.category.astype(str),
            "group": a.src_file.astype(str),
            "split": a.fold.map(
                {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
            ),
        }
    )
    write_task(
        "esc50",
        {
            "kind": "multiclass",
            "metric": "accuracy",
            "description": "Environmental sound class (ESC-50), minus clips from "
            "Freesound recordings in FSD50K or UrbanSound8K; fold 5 test, fold 4 val",
            "time_expansion": 1,
            "source": "https://github.com/karolpiczak/ESC-50",
            "license": "CC BY-NC 3.0",
            "citation": "Piczak 2015, ACM MM",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "ESC50",
        ("esc50",),
        [Remote("ESC-50-master.zip", URL, extract=True)],
        prepare,
        0.6,
        "CC BY-NC 3.0",
    )
]
