from __future__ import annotations

import json

import numpy as np

from probebench.audio import windows
from probebench.cli import main
from probebench.datasets import all_datasets, for_tasks
from probebench.paths import Paths
from probebench.tasks import discover


def test_windows_cover_the_clip():
    x = np.arange(10, dtype=np.float32)
    w = windows(x, 4, 2)
    assert w.shape == (4, 4)
    assert w[-1, -1] == 9  # the last window ends at the clip's end
    short = windows(np.ones(2, np.float32), 4, 2)
    assert short.tolist() == [[0, 1, 1, 0]]  # centered and zero-padded


def test_task_manifest_paths_are_relative(toy_task):
    (t,) = discover(Paths.resolve(toy_task).tasks)
    raw = json.loads((t.folder / "task.json").read_text())
    assert raw["kind"] == "multiclass"
    m = t.manifest()
    assert all(p.startswith(str(toy_task)) for p in m.path)
    y, classes = t.targets(m)
    assert classes == ["tone3000", "tone500"]
    assert len(y) == 60


def test_run_mel_end_to_end(toy_task, capsys):
    main(
        [
            "--data",
            str(toy_task),
            "run",
            "--model",
            "mel",
            "--no-download",
            "--workers",
            "2",
        ]
    )
    res = json.loads((toy_task / "results" / "mel_baseline.json").read_text())
    assert res["toy"]["headline"] == 1.0
    assert (toy_task / "scores" / "mel_baseline" / "toy.npz").exists()
    main(["--data", str(toy_task), "report"])
    assert "| toy | accuracy | 1.000 |" in capsys.readouterr().out


def test_every_task_has_one_dataset():
    tasks = [t for d in all_datasets() for t in d.tasks]
    assert len(tasks) == len(set(tasks)) == 15
    assert [d.name for d in for_tasks(["esc50"])] == ["ESC50"]


def test_pool_skips_non_finite_windows():
    from probebench.embed import pool

    e = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 0.0]], np.float32)
    assert pool(e).tolist() == [2.0, 1.0, 3.0, 2.0]
    assert pool(np.full((2, 2), np.nan, np.float32)).tolist() == [0.0] * 4
