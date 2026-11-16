"""Hainan gibbon calls (Dufourq et al. 2021): 13 eight-hour passive recordings
(9.6 kHz) from Bawangling with call start and end times. Each recording is cut into
10 s windows; a window is positive when a call overlaps it by >= 1 s. All positive
windows plus five times as many negatives (seeded); grouped by recording."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ..download import zenodo
from .common import Dataset, grouped_split, write_task


def prepare(raw: Path, tasks: Path) -> None:
    src = raw / "extracted"
    rng = np.random.default_rng(42)
    rows = []
    for wav in sorted(src.glob("*.wav")):
        lab = src / "Train_Labels" / f"g_{wav.stem}.data"
        calls = (
            pd.read_csv(lab) if lab.exists() else pd.DataFrame(columns=["Start", "End"])
        )
        starts = np.arange(0, sf.info(wav).duration - 10, 10.0)
        end, start = calls.End.to_numpy(float), calls.Start.to_numpy(float)
        ov = np.array(
            [
                ((np.minimum(end, s + 10) - np.maximum(start, s)) >= 1).any()
                for s in starts
            ]
        )
        pos, neg = starts[ov], starts[~ov]
        neg = rng.choice(neg, min(len(neg), 5 * max(1, len(pos))), replace=False)
        rows += [
            {
                "clip_id": f"{wav.stem}@{int(s)}",
                "path": str(wav),
                "start_s": float(s),
                "end_s": float(s + 10),
                "label": y,
                "group": wav.stem,
            }
            for s, y in [(s, 1) for s in pos] + [(s, 0) for s in neg]
        ]
    m = pd.DataFrame(rows)
    m["split"] = grouped_split(m, "label", "group")
    write_task(
        "hainan_gibbon",
        {
            "kind": "detection",
            "metric": "auc",
            "description": "Hainan gibbon call in a 10 s window of 8 h passive "
            "recordings; all positives + 5x negatives; split by recording",
            "time_expansion": 1,
            "source": "https://zenodo.org/records/3991714",
            "license": "CC BY-NC-SA 4.0",
            "citation": "Dufourq et al. 2021, Remote Sensing in Ecology and Conservation",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "Hainan_Gibbon",
        ("hainan_gibbon",),
        partial(zenodo, 3991714, keep=("Train.zip", "Train_Labels.zip")),
        prepare,
        11.9,
        "CC BY-NC-SA 4.0",
    )
]
