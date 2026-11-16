"""HumBugDB mosquito species (Kiskin et al. 2021): which species is in flight, from
field and culture recordings. Mosquito clips of species with >= 100 clips; grouped
by place and recording day. Species stays confounded with place and device for some
cultures, a property of the data. Headline mAP: macro-F1 caps near 0.5 for every
model under this split."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from ..download import zenodo
from .common import (
    Dataset,
    drop_thin_classes,
    keep_classes_in_all_splits,
    per_class_group_split,
    write_task,
)


def prepare(raw: Path, tasks: Path) -> None:
    a = pd.read_csv(raw / "neurips_2021_zenodo_0_0_1.csv")
    a = a[(a.sound_type == "mosquito") & a.species.notna()]
    counts = a.species.value_counts()
    a = a[a.species.isin(counts[counts >= 100].index)]
    day = a.record_datetime.astype(str).str.split(" ").str[0]
    m = pd.DataFrame(
        {
            "clip_id": a.id.astype(str),
            "path": [str(raw / "extracted" / f"{i}.wav") for i in a.id],
            "label": a.species.astype(str),
            "group": a.place.astype(str) + "|" + day,
        }
    )
    m, thin = drop_thin_classes(m, "label", "group")
    m["split"] = per_class_group_split(m, "label", "group")
    m, gone = keep_classes_in_all_splits(m)
    write_task(
        "humbugdb_species",
        {
            "kind": "multiclass",
            "metric": "map",
            "description": "Mosquito species in flight (HumBugDB); split by place and "
            f"day; dropped: {thin + gone}",
            "time_expansion": 1,
            "source": "https://zenodo.org/records/4904800",
            "license": "CC BY 4.0",
            "citation": "Kiskin et al. 2021, HumBugDB (NeurIPS Datasets and Benchmarks)",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "HumBugDB",
        ("humbugdb_species",),
        partial(zenodo, 4904800),
        prepare,
        4.1,
        "CC BY 4.0",
    )
]
