from __future__ import annotations

import numpy as np
import pandas as pd

from probebench.datasets.common import (
    drop_thin_classes,
    grouped_split,
    keep_classes_in_all_splits,
    per_class_group_split,
)


def _df(n_groups: int = 40, per: int = 5) -> pd.DataFrame:
    rows = [
        {"label": f"c{g % 4}", "group": f"g{g}"}
        for g in range(n_groups)
        for _ in range(per)
    ]
    return pd.DataFrame(rows)


def test_grouped_split_keeps_groups_together():
    df = _df()
    df["split"] = grouped_split(df, "label", "group")
    assert set(df.split) == {"train", "val", "test"}
    assert (df.groupby("group").split.nunique() == 1).all()


def test_per_class_group_split_balances_clips():
    df = _df()
    df["split"] = per_class_group_split(df, "label", "group")
    assert (df.groupby("group").split.nunique() == 1).all()
    for _, g in df.groupby("label"):
        frac = g.split.value_counts(normalize=True)
        assert 0.1 <= frac["test"] <= 0.3
        assert 0.1 <= frac["val"] <= 0.3


def test_thin_and_unsplit_classes_are_dropped():
    df = _df()
    df.loc[df.group.isin(["g0", "g4"]), "label"] = "rare"  # two groups only
    kept, thin = drop_thin_classes(df, "label", "group")
    assert thin == ["rare"]
    assert "rare" not in set(kept.label)
    kept["split"] = np.where(kept.label == "c1", "train", "test")
    kept, gone = keep_classes_in_all_splits(kept)
    assert "c1" in gone
