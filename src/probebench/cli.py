"""probebench command line.

    probebench tasks                            what is known, downloaded, prepared
    probebench setup [--tasks ...] [--models ...]   download + prepare (+ models)
    probebench download / prepare [--tasks ...]     the two steps separately
    probebench run --model birdnet-v3.0 [--tasks ...]   probe one model
    probebench report [--models ...]            results table (markdown)
    probebench plot [--models ...]              precision-recall figure

The data root is ``--data``, else ``PROBEBENCH_DATA``, else ``./data``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .datasets import all_datasets, for_tasks
from .download import fetch_all
from .paths import Paths
from .tasks import discover

RECIPE = (
    "windows of the model's input length, hop half a window; mean + max pooled per "
    "clip; standardized; class-balanced logistic regression, C in {0.001, 0.01, 0.1, "
    "1, 10} chosen on val; refit on train + val; test scored once"
)


def _prepared(p: Paths) -> set[str]:
    return {f.parent.name for f in p.tasks.glob("*/task.json")}


def _downloaded(p: Paths, d) -> bool:
    return (p.raw / d.name / ".complete").exists()


def download(p: Paths, tasks: list[str] | None, keep_archives: bool = False) -> None:
    for d in for_tasks(tasks):
        if _downloaded(p, d):
            continue
        print(f"== {d.name} (~{d.size_gb:g} GB, {d.license})", flush=True)
        folder = p.raw / d.name
        try:
            fetch_all(d.remotes(), folder, keep_archives=keep_archives)
        except Exception as e:  # one unreachable host must not stop the others
            print(f"{d.name}: download FAILED ({e}); rerun to resume", flush=True)
            continue
        (folder / ".complete").write_text(time.strftime("%Y-%m-%d"))


def prepare(p: Paths, tasks: list[str] | None, force: bool = False) -> None:
    done = _prepared(p)
    for d in for_tasks(tasks):
        if not force and set(d.tasks) <= done:
            continue
        if not _downloaded(p, d):
            print(f"{d.name}: not downloaded yet (probebench download)", flush=True)
            continue
        print(f"== preparing {', '.join(d.tasks)}", flush=True)
        try:
            d.prepare(p.raw / d.name, p.tasks)
        except Exception as e:
            print(f"{d.name}: prepare FAILED ({e!r})", flush=True)


def fetch_models(specs: list[str], p: Paths) -> None:
    from .models import from_spec

    for s in specs:
        a = from_spec(s, p.models)  # downloads the model file on first use
        print(f"{s}: ready ({a.name}, {a.sample_rate} Hz, {a.window_s} s)")
        if hasattr(a, "close"):
            a.close()


def cmd_tasks(a, p: Paths) -> None:
    done = _prepared(p)
    print(f"data root: {p.root}")
    for d in all_datasets():
        state = "downloaded" if _downloaded(p, d) else "-"
        for t in d.tasks:
            ready = "prepared" if t in done else state
            print(f"  {t:24s} {d.name:28s} ~{d.size_gb:>5.1f} GB  {ready}")


def cmd_setup(a, p: Paths) -> None:
    download(p, a.tasks, a.keep_archives)
    prepare(p, a.tasks)
    if a.models:
        fetch_models(a.models, p)


def cmd_download(a, p: Paths) -> None:
    download(p, a.tasks, a.keep_archives)
    if a.models:
        fetch_models(a.models, p)


def cmd_prepare(a, p: Paths) -> None:
    prepare(p, a.tasks, a.force)


def cmd_run(a, p: Paths) -> None:
    from .embed import embed_task
    from .models import from_spec
    from .probe import run_probe

    if not a.no_download:
        download(p, a.tasks)
        prepare(p, a.tasks)
    adapter = from_spec(a.model, p.models)
    out = p.results / f"{adapter.name}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    res["_meta"] = {
        "label": a.label or res.get("_meta", {}).get("label", adapter.name),
        "spec": a.model,
        "recipe": RECIPE,
        "probebench": __version__,
        **getattr(adapter, "provenance", {}),
    }
    for t in discover(p.tasks, a.tasks):
        t0 = time.time()
        m = t.manifest()
        emb = embed_task(adapter, t, p.cache, workers=a.workers)
        y, classes = t.targets(m)
        r = run_probe(t.kind, t.metric, emb, y, m.split.to_numpy(), len(classes))
        sc = p.scores / adapter.name
        sc.mkdir(parents=True, exist_ok=True)
        np.savez(
            sc / f"{t.name}.npz",
            scores=r.pop("scores"),
            y=r.pop("y"),
            kind=t.kind,
            classes=np.array(classes),
        )
        r.update(kind=t.kind, classes=len(classes), seconds=round(time.time() - t0))
        res[t.name] = r
        print(
            f"{adapter.name} {t.name}: {t.metric} {r['headline']:.3f} (C {r['C']})",
            flush=True,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    if hasattr(adapter, "close"):
        adapter.close()


def _results_dir(a, p: Paths) -> Path:
    return Path(a.results) if a.results else p.results


def cmd_report(a, p: Paths) -> None:
    from .report import load_results, table

    runs = load_results(_results_dir(a, p), a.models)
    if not runs:
        sys.exit("no results yet: probebench run --model ...")
    print(table(runs))


def cmd_plot(a, p: Paths) -> None:
    from .report import label_of, load_results, plot

    runs = load_results(_results_dir(a, p), a.models)
    models = [m for m in runs if (p.scores / m).exists()]
    out = plot(p.scores, Path(a.out), models, {m: label_of(runs[m], m) for m in models})
    print(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="probebench", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--data", help="data root (default: $PROBEBENCH_DATA or ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn, help_: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_)
        s.set_defaults(fn=fn)
        return s

    add("tasks", cmd_tasks, "list tasks and their state")
    for name, fn, h in (
        ("setup", cmd_setup, "download and prepare tasks (and models)"),
        ("download", cmd_download, "download datasets (and models)"),
    ):
        s = add(name, fn, h)
        s.add_argument("--tasks", nargs="*", help="default: all")
        s.add_argument("--models", nargs="*", default=[], help="model specs to fetch")
        s.add_argument(
            "--keep-archives",
            action="store_true",
            help="keep zip files after verified extraction",
        )
    s = add("prepare", cmd_prepare, "build task manifests from the downloads")
    s.add_argument("--tasks", nargs="*")
    s.add_argument("--force", action="store_true", help="rebuild prepared tasks")
    s = add("run", cmd_run, "embed and probe the tasks with one model")
    s.add_argument(
        "--model",
        required=True,
        help="mel, birdnet-v3.0, birdnet-v2.4, "
        "torchscript:<file.pt>[@name] or py:<module>:<factory>",
    )
    s.add_argument("--tasks", nargs="*")
    s.add_argument("--label", help="display name in reports and figures")
    s.add_argument("--workers", type=int, default=4, help="audio decoding threads")
    s.add_argument(
        "--no-download",
        action="store_true",
        help="only use tasks that are already prepared",
    )
    for name, fn, h in (
        ("report", cmd_report, "markdown results table"),
        ("plot", cmd_plot, "precision-recall figure"),
    ):
        s = add(name, fn, h)
        s.add_argument("--models", nargs="*", help="result names (default: all)")
        s.add_argument("--results", help="results folder (default: <data>/results)")
        if name == "plot":
            s.add_argument("--out", default="probebench_pr.png")
    a = ap.parse_args(argv)
    a.fn(a, Paths.resolve(a.data))


if __name__ == "__main__":
    main()
