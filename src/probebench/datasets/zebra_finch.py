"""Zebra finch call types (Elie & Theunissen 2016 vocalization library): adult and
chick calls of captive birds. Label: the call type in the file name
(<bird>_<date>-<type>-<n>.wav). Grouped by bird, so the call type has to generalize
across individuals; types with < 20 clips or missing from a split are dropped."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..download import Remote
from .common import Dataset, grouped_split, keep_classes_in_all_splits, write_task

FIGSHARE = "https://ndownloader.figshare.com/files/"


def prepare(raw: Path, tasks: Path) -> None:
    rows = []
    for p in sorted((raw / "extracted").rglob("*.wav")):
        try:
            bird, rest = p.stem.split("_", 1)
            ctype = rest.split("-")[1]
        except (ValueError, IndexError):
            continue
        rows.append({"clip_id": p.stem, "path": str(p), "label": ctype, "group": bird})
    m = pd.DataFrame(rows).drop_duplicates("clip_id")
    counts = m.label.value_counts()
    m = m[m.label.isin(counts[counts >= 20].index)].copy()
    m["split"] = grouped_split(m, "label", "group")
    m, gone = keep_classes_in_all_splits(m)
    write_task(
        "zebra_finch_calltype",
        {
            "kind": "multiclass",
            "metric": "macro_f1",
            "description": "Zebra finch call type across individuals (Elie & "
            f"Theunissen 2016); split by bird; dropped: {gone}",
            "time_expansion": 1,
            "source": "https://figshare.com/articles/dataset/11905533",
            "license": "CC BY 4.0",
            "citation": "Elie & Theunissen 2016, Animal Cognition",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "ZebraFinch_Elie_Theunissen",
        ("zebra_finch_calltype",),
        [
            Remote(
                "AdultVocalizations.zip", FIGSHARE + "21833430", 140645384, extract=True
            ),
            Remote(
                "ChickVocalizations.zip", FIGSHARE + "21833436", 50155507, extract=True
            ),
        ],
        prepare,
        0.2,
        "CC BY 4.0",
    )
]
