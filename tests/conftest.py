from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from probebench.datasets.common import write_task


@pytest.fixture
def toy_task(tmp_path):
    """A tiny multiclass task: two tones of different pitch, 30 clips each."""
    audio = tmp_path / "raw" / "toy"
    audio.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        f = 500 if i % 2 else 3000
        t = np.arange(int(1.5 * 16000)) / 16000
        x = 0.3 * np.sin(2 * np.pi * f * t) + 0.01 * rng.standard_normal(len(t))
        p = audio / f"c{i:02d}.wav"
        sf.write(p, x.astype(np.float32), 16000)
        split = "test" if i % 5 == 0 else "val" if i % 5 == 1 else "train"
        rows.append(
            {
                "clip_id": f"c{i:02d}",
                "path": str(p),
                "label": f"tone{f}",
                "group": f"g{i // 2}",
                "split": split,
            }
        )
    cfg = {"kind": "multiclass", "metric": "accuracy", "time_expansion": 1}
    write_task("toy", cfg, pd.DataFrame(rows), tmp_path / "tasks")
    return tmp_path
