"""Four BEANS-distributed datasets (Hagiwara et al. 2023), re-split by recording
group instead of BEANS' random splits, so clips of one tape, session or bout never
span train and test."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..download import Remote
from .common import (
    Dataset,
    drop_thin_classes,
    grouped_split,
    keep_classes_in_all_splits,
    per_class_group_split,
    write_task,
)

GCS = "https://storage.googleapis.com/ml-bioacoustics-datasets/"
BEANS = "Hagiwara et al. 2023 (BEANS)"


def prepare_watkins(raw: Path, tasks: Path) -> None:
    """Watkins Marine Mammal Sound Database 'best of' cuts: species. Grouped by tape
    (the first five characters of a cut's file name); species on fewer than three
    tapes are dropped. No time expansion: content above the model's band is lost,
    which is part of what the task measures."""
    src = raw / "extracted"
    a = pd.read_csv(src / "annotations.csv")
    m = pd.DataFrame(
        {
            "clip_id": [
                Path(p).stem + "_" + s for p, s in zip(a.path, a.species, strict=True)
            ],
            "path": [str(src / p) for p in a.path],
            "label": a.species.astype(str),
            "group": [Path(p).stem[:5] for p in a.path],
        }
    )
    m, thin = drop_thin_classes(m, "label", "group")
    m["split"] = per_class_group_split(m, "label", "group")
    m, unsplit = keep_classes_in_all_splits(m)
    write_task(
        "watkins",
        {
            "kind": "multiclass",
            "metric": "accuracy",
            "description": "Marine mammal species from Watkins 'best of' cuts; split by "
            f"tape; dropped (< 3 tapes): {thin + unsplit}",
            "time_expansion": 1,
            "source": GCS + "watkins.zip (BEANS)",
            "license": "Watkins Marine Mammal Sound Database: free for personal and academic use",
            "citation": f"Sayigh et al. 2016, Watkins Marine Mammal Sound Database; {BEANS}",
        },
        m,
        tasks,
    )


def prepare_dogs(raw: Path, tasks: Path) -> None:
    """Which of 10 dogs barks (Yin & McCowan 2004). Grouped by recording session
    (file name <dog>-<session>-<context>-<n>)."""
    src = raw / "extracted"
    a = pd.read_csv(src / "annotations.csv")
    m = pd.DataFrame(
        {
            "clip_id": a.filename.astype(str),
            "path": [str(src / "audio" / f) for f in a.filename],
            "label": a.name.astype(str),
            "group": a.filename.str.split("-").str[:2].str.join("-"),
        }
    )
    m, _ = drop_thin_classes(m, "label", "group")
    m["split"] = per_class_group_split(m, "label", "group")
    m, _ = keep_classes_in_all_splits(m)
    write_task(
        "dog_individual",
        {
            "kind": "multiclass",
            "metric": "accuracy",
            "description": "Which of 10 dogs is barking (Yin & McCowan 2004); split by "
            "recording session",
            "time_expansion": 1,
            "source": GCS + "dog_barks.zip (BEANS)",
            "license": "academic use (by correspondence with the authors)",
            "citation": f"Yin & McCowan 2004, Animal Behaviour; {BEANS}",
        },
        m,
        tasks,
    )


def prepare_hiceas(raw: Path, tasks: Path) -> None:
    """Minke whale 'boing' in one-minute HICEAS towed-array recordings: 1 if any
    annotation falls in the minute. Grouped by recorder and day."""
    src = raw / "extracted"
    rows = []
    for part in ("train", "valid", "test"):
        for line in (src / f"{part}.jsonl").read_text().splitlines():
            r = json.loads(line)
            name = Path(r["path"]).name
            rows.append(
                {
                    "clip_id": Path(name).stem,
                    "path": str(src / "wav" / name),
                    "label": int(bool(r["annotations"])),
                    "group": "_".join(Path(name).stem.split("_")[:2]),
                }
            )
    m = pd.DataFrame(rows).drop_duplicates("clip_id")
    m["split"] = grouped_split(m, "label", "group")
    write_task(
        "hiceas_minke",
        {
            "kind": "detection",
            "metric": "auc",
            "description": "Minke whale boing present in a one-minute HICEAS recording; "
            "split by recorder and day",
            "time_expansion": 1,
            "source": GCS + "hiceas_1-20_minke-detection.zip (BEANS)",
            "license": "NOAA (public)",
            "citation": f"Rankin et al. (HICEAS 2017); {BEANS}",
        },
        m,
        tasks,
    )


def prepare_fruit_bats(raw: Path, tasks: Path) -> None:
    """Which Egyptian fruit bat emits a social call (Prat et al. 2017), 250 kHz audio
    time-expanded 10x. File ids run in recording order, so blocks of 500 ids stand in
    for sessions; emitters with fewer than three blocks are dropped."""
    src = raw / "extracted"
    a = pd.read_csv(src / "annotations.csv")
    ids = a["File Name"].str.replace(".wav", "", regex=False).astype(int)
    m = pd.DataFrame(
        {
            "clip_id": a["File Name"].astype(str),
            "path": [str(src / "audio" / f) for f in a["File Name"]],
            "label": a.Emitter.astype(str),
            "group": (ids // 500).astype(str),
        }
    )
    m = m[m.label != "0"]  # emitter 0 = unknown
    m, thin = drop_thin_classes(m, "label", "group")
    m["split"] = per_class_group_split(m, "label", "group")
    m, gone = keep_classes_in_all_splits(m)
    write_task(
        "fruit_bat_individual",
        {
            "kind": "multiclass",
            "metric": "accuracy",
            "description": "Which Egyptian fruit bat emits a social call (Prat et al. "
            "2017), 250 kHz time-expanded 10x; split by blocks of 500 consecutive file "
            f"ids; dropped emitters: {thin + gone}",
            "time_expansion": 10,
            "source": GCS + "egyptian_fruit_bats.zip (BEANS)",
            "license": "CC BY 4.0 (Prat et al. 2017, figshare)",
            "citation": f"Prat, Azoulay, Dor & Yovel 2017, Scientific Data; {BEANS}",
        },
        m,
        tasks,
    )


def _beans(name: str, zipname: str, prepare, task: str, gb: float, lic: str) -> Dataset:
    return Dataset(
        name, (task,), [Remote(zipname, GCS + zipname, extract=True)], prepare, gb, lic
    )


DATASETS = [
    _beans(
        "Watkins_BEANS",
        "watkins.zip",
        prepare_watkins,
        "watkins",
        0.9,
        "free for personal and academic use",
    ),
    _beans(
        "Dog_Barks_BEANS",
        "dog_barks.zip",
        prepare_dogs,
        "dog_individual",
        0.9,
        "academic use",
    ),
    _beans(
        "HICEAS_Minke_BEANS",
        "hiceas_1-20_minke-detection.zip",
        prepare_hiceas,
        "hiceas_minke",
        1.4,
        "NOAA (public)",
    ),
    _beans(
        "Egyptian_Fruit_Bats_BEANS",
        "egyptian_fruit_bats.zip",
        prepare_fruit_bats,
        "fruit_bat_individual",
        5.2,
        "CC BY 4.0",
    ),
]
