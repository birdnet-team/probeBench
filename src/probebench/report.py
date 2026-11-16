"""Results table and precision-recall figure from ``<root>/results`` and
``<root>/scores``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Categorical slots in fixed order (validated for color-vision deficiency on a light
# surface); the mel baseline is always the gray dashed reference.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
BASELINE = ("#52514e", "--")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
R = np.linspace(0, 1, 101)


def load_results(folder: Path, models: list[str] | None = None) -> dict[str, dict]:
    files = sorted(folder.glob("*.json"))
    runs = {f.stem: json.loads(f.read_text()) for f in files}
    if models:
        runs = {m: runs[m] for m in models if m in runs}
    return runs


def label_of(run: dict, name: str) -> str:
    return run.get("_meta", {}).get("label", name)


def table(runs: dict[str, dict]) -> str:
    """Markdown table: headline metric per task and model, plus the mean over the
    tasks every model has (and over all of each model's tasks when they differ)."""
    tasks = sorted({t for r in runs.values() for t in r if not t.startswith("_")})
    names = list(runs)
    head = [label_of(runs[n], n) for n in names]
    lines = [
        "| Task | Metric | " + " | ".join(head) + " |",
        "|---|---|" + "---:|" * len(names),
    ]
    for t in tasks:
        metric = next(r[t]["metric"] for r in runs.values() if t in r)
        cells = [
            f"{runs[n][t]['headline']:.3f}" if t in runs[n] else "-" for n in names
        ]
        lines.append(f"| {t} | {metric} | " + " | ".join(cells) + " |")
    common = [t for t in tasks if all(t in runs[n] for n in names)]
    means = [np.mean([runs[n][t]["headline"] for t in common]) for n in names]
    lines.append(
        f"| **Mean ({len(common)} tasks)** | | "
        + " | ".join(f"**{m:.3f}**" for m in means)
        + " |"
    )
    if len(common) < len(tasks):
        own = [
            f"{np.mean([v['headline'] for k, v in runs[n].items() if not k.startswith('_')]):.3f}"
            for n in names
        ]
        lines.append("| Mean (all of a model's tasks) | | " + " | ".join(own) + " |")
    return "\n".join(lines)


def interp_pr(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    """Interpolated precision at each recall in R (the best precision at recall >= r)
    and average precision (scikit-learn's step sum)."""
    order = np.argsort(-s, kind="stable")
    y = y[order].astype(bool)
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / max(1, y.sum())
    ap = float(np.sum(np.diff(np.r_[0, rec]) * prec))
    env = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, R, side="left")
    p = np.where(idx < len(env), env[np.minimum(idx, len(env) - 1)], 0.0)
    return p, ap


def task_curve(f: Path) -> tuple[np.ndarray, float]:
    """Macro PR curve of one task: one-vs-rest per class with test positives."""
    z = np.load(f, allow_pickle=False)
    s, y, kind = z["scores"], z["y"], str(z["kind"])
    if kind == "detection":
        return interp_pr(y, s)
    if kind == "multiclass":
        y = np.eye(s.shape[1], dtype=np.int8)[y]
    curves, aps = [], []
    for k in range(s.shape[1]):
        if y[:, k].any():
            p, ap = interp_pr(y[:, k], s[:, k])
            curves.append(p)
            aps.append(ap)
    return np.mean(curves, 0), float(np.mean(aps))


def plot(
    scores_root: Path, out: Path, models: list[str], labels: dict[str, str]
) -> Path:
    """Mean macro PR curve over the tasks all models have, plus one panel per task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = {m: {p.stem for p in (scores_root / m).glob("*.npz")} for m in models}
    tasks = sorted(set.intersection(*have.values()))
    if not tasks:
        raise ValueError("no task has scores for every model")
    curves = {
        m: {t: task_curve(scores_root / m / f"{t}.npz") for t in tasks} for m in models
    }
    learned = [m for m in models if not m.startswith("mel")]
    if len(learned) > len(COLORS):
        raise ValueError(
            f"at most {len(COLORS)} models (plus the mel baseline) per figure"
        )
    style = {}
    slot = iter(COLORS)
    for m in models:
        style[m] = BASELINE if m.startswith("mel") else (next(slot), "-")

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    cols = 4
    rows = -(-(len(tasks) + 1) // cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 2.8 * rows),
        facecolor=SURFACE,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    panels = [(f"Mean over {len(tasks)} tasks", None)] + [(t, t) for t in tasks]
    for ax, (title, t) in zip(axes, panels, strict=False):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.6)
        for m in models:
            color, ls = style[m]
            if t is None:
                p = np.mean([curves[m][x][0] for x in tasks], 0)
                apv = float(np.mean([curves[m][x][1] for x in tasks]))
                ax.plot(
                    R,
                    p,
                    color=color,
                    ls=ls,
                    lw=2.2,
                    label=f"{labels[m]}  mAP {apv:.3f}",
                )
            else:
                ax.plot(R, curves[m][t][0], color=color, ls=ls, lw=1.6)
        bold = "bold" if t is None else None
        ax.set_title(title, fontsize=9, color=INK, loc="left", fontweight=bold)
        if t is None:
            ax.legend(loc="lower left", fontsize=7, frameon=False)
        else:
            txt = "  ".join(f"{curves[m][t][1]:.2f}" for m in models)
            ax.text(
                0.98,
                0.04,
                "mAP " + txt,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=INK2,
                transform=ax.transAxes,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    for ax in axes[::cols]:
        ax.set_ylabel("Precision")
    for ax in axes[-cols:]:
        ax.set_xlabel("Recall")
    fig.suptitle(
        "probeBench: linear probes on frozen embeddings, macro PR per task (test "
        "splits); per-panel mAP in legend order",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
