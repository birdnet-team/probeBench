"""Shared pieces of the dataset modules: grouped stratified splits and writing a task
folder."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..download import Remote


@dataclass(frozen=True)
class Dataset:
    """A source dataset: where to get it and how to turn it into tasks.

    name     folder under ``<root>/raw``
    tasks    task names ``prepare`` writes
    files    the files to download (or a function listing them, e.g. a Zenodo record)
    prepare  ``prepare(raw_dir, tasks_root)`` writes the task folders
    size_gb  approximate download size
    """

    name: str
    tasks: tuple[str, ...]
    files: list[Remote] | Callable[[], list[Remote]]
    prepare: Callable[[Path, Path], None]
    size_gb: float
    license: str = ""

    def remotes(self) -> list[Remote]:
        return self.files() if callable(self.files) else list(self.files)


def grouped_split(
    df: pd.DataFrame, label: str, group: str, seed: int = 42, folds: int = 5
) -> pd.Series:
    """train / val / test (3:1:1 by groups) with every group in one split and class
    proportions kept as far as the groups allow (StratifiedGroupKFold: fold 0 test,
    fold 1 val, the rest train)."""
    y = df[label].astype(str).to_numpy()
    g = df[group].astype(str).to_numpy()
    split = np.empty(len(df), dtype=object)
    kf = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(kf.split(y, y, g)):
        split[idx] = "test" if k == 0 else "val" if k == 1 else "train"
    return pd.Series(split, index=df.index)


def per_class_group_split(
    df: pd.DataFrame, label: str, group: str, seed: int = 42, frac: float = 0.2
) -> pd.Series:
    """For data whose groups (tapes, sessions) mostly hold one class: per class,
    shuffle its groups and fill test, then val, to about ``frac`` of the class's clips
    each (at least one group each); the rest is train. Balanced by clip counts, so a
    few large groups cannot swallow a split. A group with several classes goes with
    its majority class. Needs >= 3 groups per class (see ``drop_thin_classes``)."""
    rng = np.random.default_rng(seed)
    major = df.groupby(group)[label].agg(
        lambda s: s.astype(str).value_counts().index[0]
    )
    size = df.groupby(group).size()
    where = {}
    for _, gs in major.groupby(major):
        gs = list(gs.index)
        rng.shuffle(gs)
        cap = frac * size[gs].sum()
        filled = {"test": 0, "val": 0}
        for g in gs:
            target = next(
                (t for t in ("test", "val") if filled[t] + size[g] <= cap), "train"
            )
            if target != "train":
                filled[target] += size[g]
            where[g] = target
        for t in ("test", "val"):  # every split gets a group: the smallest train one
            if filled[t] == 0:
                train = [g for g in gs if where[g] == "train"]
                if len(train) > 1:
                    where[min(train, key=lambda g: size[g])] = t
    return df[group].map(where)


def keep_classes_in_all_splits(
    df: pd.DataFrame, label: str = "label"
) -> tuple[pd.DataFrame, list]:
    """Drop classes that the grouped split left without train, val or test clips."""
    present = df.groupby(label).split.nunique()
    gone = sorted(present[present < 3].index.astype(str))
    return df[~df[label].astype(str).isin(gone)].copy(), gone


def drop_thin_classes(
    df: pd.DataFrame, label: str, group: str, min_groups: int = 3
) -> tuple[pd.DataFrame, list]:
    """Classes recorded in fewer than ``min_groups`` groups cannot appear in train, val
    and test; drop them."""
    n = df.groupby(label)[group].nunique()
    thin = sorted(n[n < min_groups].index.astype(str))
    return df[~df[label].astype(str).isin(thin)].copy(), thin


def write_task(name: str, cfg: dict, manifest: pd.DataFrame, tasks_root: Path) -> Path:
    """Write ``<tasks_root>/<name>/task.json`` and ``manifest.parquet`` (audio paths
    relative to the task folder, so the data root can move) and print a summary."""
    d = tasks_root / name
    d.mkdir(parents=True, exist_ok=True)
    m = manifest.reset_index(drop=True).copy()
    for c in ("start_s", "end_s"):
        if c not in m:
            m[c] = 0.0 if c == "start_s" else np.nan
    m["path"] = [os.path.relpath(p, d) for p in m.path]
    cols = ["clip_id", "path", "start_s", "end_s", "label", "split", "group"]
    m = m[cols + (["time_expansion"] if "time_expansion" in m else [])]
    if m.clip_id.duplicated().any():
        raise ValueError(f"{name}: duplicate clip ids")
    m.to_parquet(d / "manifest.parquet", index=False)
    (d / "task.json").write_text(json.dumps({"name": name, **cfg}, indent=1))
    by = m.groupby("split").size().to_dict()
    lab = m.label.astype(str) if cfg["kind"] != "multilabel" else m.label.map(len)
    print(
        f"{name}: {len(m)} clips, {m.group.nunique()} groups, "
        f"{lab.nunique()} label values, splits {by}"
    )
    return d
