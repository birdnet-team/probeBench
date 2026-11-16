"""The source datasets and the tasks built from them.

Each module defines ``DATASETS``: a list of :class:`~.common.Dataset` (where to
download it, the tasks it yields and a ``prepare(raw_dir, tasks_root)`` function).
To add a task, add a module and list it in ``MODULES``.
"""

from __future__ import annotations

import importlib

from .common import Dataset

MODULES = (
    "beans",
    "zebra_finch",
    "humbugdb",
    "insectset66",
    "chiffchaff",
    "anafcc",
    "edansa",
    "hainan_gibbon",
    "chiroset",
    "esc50",
    "right_whale",
)


def all_datasets() -> list[Dataset]:
    out: list[Dataset] = []
    for name in MODULES:
        out += importlib.import_module(f"{__name__}.{name}").DATASETS
    return out


def for_tasks(tasks: list[str] | None) -> list[Dataset]:
    """The datasets needed for ``tasks`` (all when None)."""
    ds = all_datasets()
    if not tasks:
        return ds
    known = {t for d in ds for t in d.tasks}
    unknown = sorted(set(tasks) - known)
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; known: {sorted(known)}")
    return [d for d in ds if set(d.tasks) & set(tasks)]
