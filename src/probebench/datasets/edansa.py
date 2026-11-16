"""EDANSA-2019 (Coban et al. 2022): Arctic North Slope (Alaska) soundscapes, 10 s
clips, multi-label tags (biophony, anthrophony, geophony, silence and finer tags:
songbird, aircraft, insects, wind, rain, ...). The dataset's own train / valid /
test sets (split by site); tags with >= 30 positives in train and >= 10 in each of
valid and test are kept; scored by macro mAP over tags."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from ..download import zenodo
from .common import Dataset, write_task


def prepare(raw: Path, tasks: Path) -> None:
    src = raw / "extracted" / "EDANSA-2019"
    a = pd.read_csv(src / "labels.csv")
    tags = list(a.columns[a.columns.get_loc("Anth") :])
    split = a["set"].map({"train": "train", "valid": "val", "test": "test"})
    keep = [
        t
        for t in tags
        if a.loc[split == "train", t].sum() >= 30
        and a.loc[split == "val", t].sum() >= 10
        and a.loc[split == "test", t].sum() >= 10
    ]
    m = pd.DataFrame(
        {
            "clip_id": a["Clip Path"].astype(str),
            "path": [str(src / "data" / p) for p in a["Clip Path"]],
            "label": [[t for t in keep if r[t] == 1] for _, r in a[keep].iterrows()],
            "group": a.region.astype(str) + "|" + a["Site ID"].astype(str),
            "split": split,
        }
    )
    m = m[m.split.notna()]
    write_task(
        "edansa_arctic",
        {
            "kind": "multilabel",
            "metric": "map",
            "description": f"Arctic soundscape tags (EDANSA-2019), official site split; tags: {keep}",
            "time_expansion": 1,
            "source": "https://zenodo.org/records/6824272",
            "license": "CC BY 4.0",
            "citation": "Coban, Perra & Mandel 2022, EDANSA-2019",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "EDANSA_2019",
        ("edansa_arctic",),
        partial(zenodo, 6824272),
        prepare,
        7.9,
        "CC BY 4.0",
    )
]
