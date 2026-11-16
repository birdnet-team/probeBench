"""InsectSet66 (Faiss & Stowell 2023): Orthoptera and cicada species, with the
dataset's official split (a source recording stays in one subset)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from ..download import zenodo
from .common import Dataset, write_task


def prepare(raw: Path, tasks: Path) -> None:
    a = pd.read_csv(raw / "InsectSet66_Train_Val_Test_Annotation.csv")
    root = raw / "extracted" / "InsectSet66_Train_Val_Test"
    files = {p.name: p for p in root.rglob("*.wav")}
    a = a[a.file_name.isin(files)]
    m = pd.DataFrame(
        {
            "clip_id": a.file_name.astype(str),
            "path": [str(files[f]) for f in a.file_name],
            "label": a.species.astype(str),
            "group": a.unique_file.astype(str),
            "split": a.subset.map(
                {"train": "train", "validation": "val", "test": "test"}
            ),
        }
    )
    write_task(
        "insectset66",
        {
            "kind": "multiclass",
            "metric": "accuracy",
            "description": "Orthoptera and cicada species (InsectSet66), official split",
            "time_expansion": 1,
            "source": "https://zenodo.org/records/8252141",
            "license": "CC BY 4.0",
            "citation": "Faiss & Stowell 2023, InsectSet47/66",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "InsectSet66",
        ("insectset66",),
        partial(zenodo, 8252141, keep=("InsectSet66",)),
        prepare,
        5.2,
        "CC BY 4.0",
    )
]
