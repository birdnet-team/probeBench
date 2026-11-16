"""The linear probe: standardized embeddings, class-balanced logistic regression,
C chosen on the validation split, refit on train + val, scored once on test."""

from __future__ import annotations

import contextlib
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
N_JOBS = int(os.environ.get("PROBEBENCH_JOBS", 4))


def _model(kind: str, c: float):
    lr = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
    return OneVsRestClassifier(lr, n_jobs=N_JOBS) if kind == "multilabel" else lr


def _scores(kind: str, clf, x: np.ndarray, n_classes: int) -> np.ndarray:
    p = clf.predict_proba(x)
    if kind == "detection":
        return p[:, 1]
    if kind == "multiclass" and p.shape[1] != n_classes:
        full = np.zeros((len(x), n_classes))  # classes absent from the fit score 0
        full[:, clf.classes_] = p
        return full
    return p


def metrics(kind: str, y: np.ndarray, s: np.ndarray, n_classes: int = 0) -> dict:
    """accuracy / macro_f1 / map / auc, where each is defined for the task kind."""
    out = {}
    if kind == "multiclass":
        pred = s.argmax(1)
        out["accuracy"] = accuracy_score(y, pred)
        out["macro_f1"] = f1_score(
            y, pred, average="macro", labels=np.unique(y), zero_division=0
        )
        onehot = np.eye(n_classes)[y]
        keep = onehot.sum(0) > 0
        out["map"] = average_precision_score(
            onehot[:, keep], s[:, keep], average="macro"
        )
    elif kind == "multilabel":
        keep = y.sum(0) > 0
        out["map"] = average_precision_score(y[:, keep], s[:, keep], average="macro")
        out["macro_f1"] = f1_score(
            y[:, keep],
            (s[:, keep] >= 0.5).astype(int),
            average="macro",
            zero_division=0,
        )
        with contextlib.suppress(ValueError):  # a class without negatives in test
            out["auc"] = roc_auc_score(y[:, keep], s[:, keep], average="macro")
    else:
        out["auc"] = roc_auc_score(y, s)
        out["map"] = average_precision_score(y, s)
        out["accuracy"] = accuracy_score(y, (s >= 0.5).astype(int))
    return {k: round(float(v), 4) for k, v in out.items()}


def run_probe(
    kind: str,
    metric: str,
    emb: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    n_classes: int,
    c_grid: tuple[float, ...] = C_GRID,
) -> dict:
    """Probe one task. Returns the headline metric, all test metrics, the chosen C,
    split sizes, and the test scores and labels (``scores``, ``y``)."""
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    x = StandardScaler().fit(emb[tr]).transform(emb).astype(np.float32)
    best = (-np.inf, c_grid[0])
    for c in c_grid:
        clf = _model(kind, c).fit(x[tr], y[tr])
        s = _scores(kind, clf, x[va], n_classes)
        v = metrics(kind, y[va], s, n_classes).get(metric, -np.inf)
        if v > best[0]:
            best = (v, c)
    trva = tr | va
    x = StandardScaler().fit(emb[trva]).transform(emb).astype(np.float32)
    clf = _model(kind, best[1]).fit(x[trva], y[trva])
    s_te = _scores(kind, clf, x[te], n_classes)
    res = metrics(kind, y[te], s_te, n_classes)
    return {
        "headline": res.get(metric),
        "metric": metric,
        "C": best[1],
        "val": round(float(best[0]), 4),
        "test": res,
        "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "scores": s_te.astype(np.float32),
        "y": np.asarray(y[te]),
    }
