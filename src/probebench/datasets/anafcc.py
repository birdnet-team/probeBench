"""BirdVox-ANAFCC (Cramer et al. 2020): night flight calls of 14 northeastern
American passerines from BirdVox-70k, CLO-43SD / CLO-SWTH / CLO-WTSP and Old Bird
clips (the dataset's Xeno-canto and Macaulay files are not downloaded).

Species-level taxonomy codes only; at most 1,500 clips per species (seeded). Clips
are exported from the HDF5 files to 16-bit WAV under wav/. Grouped by recording unit
and a coarse time block of the clip key, so calls of one night stretch never span
splits."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf

from ..download import zenodo
from .common import (
    Dataset,
    drop_thin_classes,
    keep_classes_in_all_splits,
    per_class_group_split,
    write_task,
)


def prepare(raw: Path, tasks: Path) -> None:
    out = raw / "wav"
    rows = []
    for f in sorted(raw.glob("*.h5")):
        if not (raw / f".{f.name}.done").exists():
            continue
        code = f.name.split("_")[-2]
        fam, sp = code.split("-")[1:]
        if not code.startswith("1-") or sp in ("0", "X") or fam == "X":
            continue
        unit = f.name.rsplit("_", 2)[0]
        with h5py.File(f, "r") as h:
            for k in h["waveforms"]:
                tok = k.split("_")
                # BirdVox-70k keys: unitNN_<time offset>_...: a coarse time block;
                # CLO / Old Bird keys: <SPECIES>_<recording id>_...: the recording
                if unit.startswith("BirdVox") and len(tok) > 1:
                    group = f"{unit}|{tok[1][:3]}"
                else:
                    group = f"{unit}|{tok[1] if len(tok) > 1 else k}"
                rows.append(
                    {
                        "file": str(f),
                        "key": k,
                        "label": code,
                        "unit": unit,
                        "group": group,
                    }
                )
    d = pd.DataFrame(rows)
    d = d.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 1500), random_state=42)
    )
    pmap = {}
    for f, g in d.groupby("file"):
        with h5py.File(f, "r") as h:
            sr = int(h["sample_rate"][()])
            for k in g.key:
                p = out / g.label.iloc[0] / f"{g.unit.iloc[0]}__{k}.wav"
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    x = np.asarray(h["waveforms"][k], np.float32)
                    x = np.clip(x / max(1e-9, np.abs(x).max()) * 0.9, -1, 1)
                    sf.write(p, x, sr, subtype="PCM_16")
                pmap[(f, k)] = str(p)
    m = pd.DataFrame(
        {
            "clip_id": d.unit + "__" + d.key,
            "path": [pmap[(f, k)] for f, k in zip(d.file, d.key, strict=True)],
            "label": d.label,
            "group": d.group,
        }
    )
    m, thin = drop_thin_classes(m, "label", "group")
    m["split"] = per_class_group_split(m, "label", "group")
    m, gone = keep_classes_in_all_splits(m)
    write_task(
        "anafcc_flightcalls",
        {
            "kind": "multiclass",
            "metric": "macro_f1",
            "description": "Night flight-call species (14 northeastern American "
            "passerines, BirdVox-ANAFCC), <= 1,500 clips per species; split by unit "
            f"and time block; dropped: {thin + gone}",
            "time_expansion": 1,
            "source": "https://zenodo.org/records/5950000 (without the xeno-canto* "
            "and macaulay* files)",
            "license": "CC BY 4.0",
            "citation": "Cramer, Lostanlen, Farnsworth, Salamon & Bello 2020, ICASSP",
        },
        m,
        tasks,
    )


DATASETS = [
    Dataset(
        "BirdVox_ANAFCC",
        ("anafcc_flightcalls",),
        partial(zenodo, 5950000, drop=("xeno-canto", "macaulay")),
        prepare,
        5.7,
        "CC BY 4.0",
    )
]
