"""The mel baseline: mean and standard deviation over time of a log-mel
spectrogram. No learning; the floor any learned embedding should beat. Also the
shortest complete example of an adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import register


class MelBaseline:
    name = "mel_baseline"
    sample_rate = 24000
    window_s = 2.5

    def __init__(self) -> None:
        import librosa

        self._fb = librosa.filters.mel(
            sr=self.sample_rate, n_fft=1024, n_mels=128, fmin=50, fmax=12000
        )

    def embed(self, x: np.ndarray) -> np.ndarray:
        import librosa

        out = []
        for w in x:
            s = np.abs(librosa.stft(w, n_fft=1024, hop_length=256)) ** 2
            m = np.log(self._fb @ s + 1e-6)
            out.append(np.concatenate([m.mean(1), m.std(1)]))
        return np.stack(out).astype(np.float32)


@register("mel")
def make(models_dir: Path) -> MelBaseline:
    return MelBaseline()
