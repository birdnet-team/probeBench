from __future__ import annotations

import numpy as np

from probebench.probe import metrics, run_probe


def _data(kind: str, n: int = 300, d: int = 16, k: int = 3):
    rng = np.random.default_rng(1)
    if kind == "multiclass":
        y = rng.integers(0, k, n)
        x = rng.standard_normal((n, d)) + 2 * np.eye(k, d)[y]
    elif kind == "multilabel":
        y = (rng.random((n, k)) < 0.4).astype(np.int8)
        x = rng.standard_normal((n, d)) + 2 * y @ np.eye(k, d)
    else:
        y = rng.integers(0, 2, n)
        x = rng.standard_normal((n, d)) + 2 * y[:, None] * np.eye(1, d)
    split = np.array(["train", "train", "train", "val", "test"] * (n // 5))
    return x.astype(np.float32), y, split


def test_separable_tasks_score_high():
    for kind, metric in (
        ("multiclass", "accuracy"),
        ("multilabel", "map"),
        ("detection", "auc"),
    ):
        x, y, split = _data(kind)
        r = run_probe(kind, metric, x, y, split, 3 if kind != "detection" else 2)
        assert r["headline"] > 0.8, (kind, r["headline"])
        assert r["C"] in (0.001, 0.01, 0.1, 1.0, 10.0)
        assert r["n"] == {"train": 180, "val": 60, "test": 60}
        assert len(r["scores"]) == 60


def test_multiclass_metrics_perfect_and_absent_class():
    y = np.array([0, 1, 1, 0])
    s = np.array([[0.9, 0.1, 0], [0.2, 0.8, 0], [0.1, 0.9, 0], [0.7, 0.3, 0]])
    m = metrics("multiclass", y, s, 3)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["map"] == 1.0  # class 2 has no test clips and is left out


def test_detection_metrics():
    m = metrics("detection", np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]))
    assert m["auc"] == 0.75
    assert m["accuracy"] == 0.5
