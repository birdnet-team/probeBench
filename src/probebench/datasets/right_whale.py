"""North Atlantic right-whale upcall detection (Marinexplore and Cornell whale
detection challenge 2013, the MONSTER version on Hugging Face): 30,000 two-second
hydrophone clips at 2 kHz, label 1 = upcall. Clips are written as 2 kHz WAV files
(wav/). Test = MONSTER fold 0 (stratified random); val = a seeded, stratified fifth
of the rest. The data carry no recording or buoy ids, so the split is not grouped."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ..download import Remote
from .common import Dataset, write_task

HF = (
    "https://huggingface.co/datasets/monster-monash/CornellWhaleChallenge/resolve/main/"
)
FILES = [
    "CornellWhaleChallenge_X.npy",
    "CornellWhaleChallenge_y.npy",
    "test_indices_fold_0.txt",
]


def prepare(raw: Path, tasks: Path) -> None:
    x_all = np.load(raw / FILES[0], mmap_mode="r")
    y = np.load(raw / FILES[1]).astype(int)
    test = {int(i) for i in (raw / FILES[2]).read_text().split()}
    out = raw / "wav"
    out.mkdir(exist_ok=True)
    ids = [f"rw_{i:05d}" for i in range(len(x_all))]
    for i, cid in enumerate(ids):
        p = out / f"{cid}.wav"
        if not p.exists():
            x = np.asarray(x_all[i], np.float32).ravel()
            sf.write(p, x / max(1e-9, np.abs(x).max()) * 0.9, 2000, subtype="PCM_16")
    rng = np.random.default_rng(42)
    split = np.array(["test" if i in test else "train" for i in range(len(y))], object)
    for c in (0, 1):
        idx = np.flatnonzero((split == "train") & (y == c))
        split[rng.choice(idx, len(idx) // 5, replace=False)] = "val"
    m = pd.DataFrame(
        {
            "clip_id": ids,
            "path": [str(out / f"{c}.wav") for c in ids],
            "label": y,
            "group": ids,
            "split": split,
        }
    )
    write_task(
        "right_whale_upcall",
        {
            "kind": "detection",
            "metric": "auc",
            "description": "Right-whale upcall in a 2 s hydrophone clip (2 kHz); "
            "MONSTER fold 0 as test, ungrouped",
            "time_expansion": 1,
            "source": "https://huggingface.co/datasets/monster-monash/CornellWhaleChallenge",
            "license": "Cornell University (challenge data)",
            "citation": "Karpistsenko, Spalding & Cukierski 2013, Marinexplore and "
            "Cornell University whale detection challenge",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "RightWhale_CornellChallenge",
        ("right_whale_upcall",),
        [Remote(f, HF + f) for f in FILES],
        prepare,
        0.5,
        "Cornell University",
    )
]
