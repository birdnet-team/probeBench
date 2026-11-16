"""ChirosetEurope (2026): European bat species from full-spectrum echolocation
recordings, the recordings hosted in the dataset's audio.zip only (link-only rows
from other platforms are not fetched). Pre-expanded recordings (factor 10) are used
as they are; native recordings at >= 192 kHz are time-expanded 10x on load, so
20-120 kHz calls land at 2-12 kHz either way; native recordings below 192 kHz are
dropped. The dataset's own grouped split (source, place and date) is kept."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from ..download import zenodo
from .common import Dataset, keep_classes_in_all_splits, write_task


def prepare(raw: Path, tasks: Path) -> None:
    a = pd.read_csv(raw / "ChirosetEurope_manifest_v1.csv", low_memory=False)
    files = {p.name: p for p in (raw / "extracted").rglob("*.wav")}
    a = a[a.file_name.isin(files)].copy()
    factor = a.time_expansion_factor.astype(float)
    already = factor >= 10
    native_ok = (factor == 1) & (a.sampling_rate.astype(float) >= 192000)
    a = a[already | native_ok]
    m = pd.DataFrame(
        {
            "clip_id": a.chiroset_id.astype(str),
            "path": [str(files[f]) for f in a.file_name],
            "label": a.latin.astype(str),
            "group": a.group_key.astype(str),
            "split": a.split.map({"train": "train", "val": "val", "test": "test"}),
            "time_expansion": [
                1.0 if t >= 10 else 10.0 for t in a.time_expansion_factor.astype(float)
            ],
        }
    )
    m, gone = keep_classes_in_all_splits(m)
    write_task(
        "chiroset_bats",
        {
            "kind": "multiclass",
            "metric": "macro_f1",
            "description": "European bat species from echolocation calls "
            "(ChirosetEurope, hosted recordings); native >= 192 kHz recordings "
            "time-expanded x10, pre-expanded ones as they are; official grouped "
            f"split; dropped: {gone}",
            "time_expansion": 10,
            "source": "https://zenodo.org/records/20773227",
            "license": "per recording (mostly CC BY-NC-SA / CC BY-ND); dataset CC BY-ND",
            "citation": "ChirosetEurope v1 (2026)",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "ChirosetEurope",
        ("chiroset_bats",),
        partial(zenodo, 20773227, keep=("ChirosetEurope_manifest_v1.csv", "audio.zip")),
        prepare,
        9.6,
        "CC BY-ND (per recording licenses)",
    )
]
