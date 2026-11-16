"""Chiffchaff individual identification (Stowell, Petruskova, Salek & Linhart 2019),
Czech field recordings, foreground calls, the dataset's official train/test lists:

  chiffchaff_withinyear  13 males, train and test from the same season
  chiffchaff_acrossyear  10 males, tested in a later season

Validation (only for choosing C) is a seeded, stratified fifth of the training list:
many males have one recording day in training, so a day-grouped validation split
would remove them from training altogether."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from ..download import zenodo
from .common import Dataset, write_task


def _load(csv: Path) -> pd.DataFrame:
    a = pd.read_csv(csv)
    ind = a.columns[1:]
    lab = a[ind].fillna(0).to_numpy().argmax(1)
    return pd.DataFrame({"file": a.wavfilename, "label": [ind[i] for i in lab]})


def prepare(raw: Path, tasks: Path) -> None:
    src = raw / "extracted"
    wavs = {p.name: p for p in src.rglob("*.wav")}
    for name, variant, desc in (
        ("chiffchaff_withinyear", "withinyear", "same season"),
        ("chiffchaff_acrossyear", "acrossyear", "test in a later season"),
    ):
        trn = _load(src / "csv" / f"chiffchaff-{variant}-fg-trn.csv")
        tst = _load(src / "csv" / f"chiffchaff-{variant}-fg-tst.csv")
        day = trn.file.str.extract(r"(day\d+)", expand=False).fillna("day0")
        trn["group"] = day + "|" + trn.label
        val_idx = set(
            trn.groupby("label", group_keys=False)
            .sample(frac=0.2, random_state=42)
            .index
        )
        trn["split"] = ["val" if i in val_idx else "train" for i in trn.index]
        tst["group"] = "test|" + tst.label
        tst["split"] = "test"
        d = pd.concat([trn, tst], ignore_index=True)
        d = d[d.file.isin(wavs)]
        m = pd.DataFrame(
            {
                "clip_id": d.split + "|" + d.file,
                "path": [str(wavs[f]) for f in d.file],
                "label": d.label,
                "group": d.group,
                "split": d.split,
            }
        )
        write_task(
            name,
            {
                "kind": "multiclass",
                "metric": "accuracy",
                "description": f"Which chiffchaff male is calling ({desc}); official "
                "train/test lists, validation = stratified 20% of train",
                "time_expansion": 1,
                "source": "https://zenodo.org/records/1413495",
                "license": "CC BY 4.0 (copyright held by the recordists)",
                "citation": "Stowell, Petruskova, Salek & Linhart 2019, J. R. Soc. Interface",
            },
            m,
            tasks,
        )


DATASETS = [
    Dataset(
        "Chiffchaff_Stowell",
        ("chiffchaff_withinyear", "chiffchaff_acrossyear"),
        partial(zenodo, 1413495, keep=("csv.zip", "chiffchaff-fg.zip")),
        prepare,
        2.7,
        "CC BY 4.0",
    )
]
