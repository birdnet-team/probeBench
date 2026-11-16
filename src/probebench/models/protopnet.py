"""ConvNeXt encoders with a prototype head, probed at the encoder: the global average
of the ConvNeXt feature map that the prototype layer compares with its prototypes
(1,024 values), not the prototype similarities or class scores, so the probe sees
the same kind of representation as for the other models.

    audioprotopnet-20  AudioProtoPNet-20 BirdSet XCL (Heinrich et al. 2025),
                       Hugging Face DBD-research-group/AudioProtoPNet-20-BirdSet-XCL
                       (transformers, the repository's own code), 32 kHz, 5 s
    birdcode           BirdCODE (Earth Species Project), Hugging Face
                       EarthSpeciesProject/sed-birdcode (package sound_event_detection,
                       Python >= 3.12), 32 kHz, 5 s; MIT

Both run on a CUDA GPU with fp16 autocast and small batches
(``PROBEBENCH_TORCH_BATCH``, default 2, halved on out-of-memory); the log-mel
stays in fp32, where loud audio would overflow fp16.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import register

APN_REPO = "DBD-research-group/AudioProtoPNet-20-BirdSet-XCL"
BIRDCODE_REPO = "EarthSpeciesProject/sed-birdcode"


def _hf_cache(models_dir: Path) -> None:
    """Hugging Face downloads go to <data>/models/huggingface unless HF_HUB_CACHE is set."""
    os.environ.setdefault("HF_HUB_CACHE", str(models_dir / "huggingface"))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "these models need a CUDA GPU (they are too heavy for a CPU run)"
        )
    frac = float(os.environ.get("PROBEBENCH_GPU_FRACTION", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    return torch


class _ConvNeXtPrototypeEncoder:
    sample_rate = 32000
    window_s = 5.0

    def __init__(self) -> None:
        self.torch = _torch()
        self.batch = int(os.environ.get("PROBEBENCH_TORCH_BATCH", 2))

    def _features(self, x):  # [B, samples] tensor on the GPU -> [B, C, H, W]
        raise NotImplementedError

    def embed(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        out, i = [], 0
        while i < len(x):
            b = torch.from_numpy(
                np.ascontiguousarray(x[i : i + self.batch], np.float32)
            )
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    f = self._features(b.cuda())
                out.append(f.float().mean(dim=(2, 3)).cpu().numpy())
                i += len(b)
            except torch.OutOfMemoryError:
                if self.batch == 1:
                    raise
                torch.cuda.empty_cache()
                self.batch //= 2
        return np.concatenate(out)


class AudioProtoPNet(_ConvNeXtPrototypeEncoder):
    name = "audioprotopnet20_xcl"

    def __init__(self) -> None:
        super().__init__()
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForSequenceClassification,
        )

        self.fe = AutoFeatureExtractor.from_pretrained(APN_REPO, trust_remote_code=True)
        self.m = (
            AutoModelForSequenceClassification.from_pretrained(
                APN_REPO, trust_remote_code=True
            )
            .cuda()
            .eval()
        )
        self.provenance = {"source": f"huggingface {APN_REPO}", "layer": "encoder GAP"}

    def _features(self, x):
        with self.torch.autocast("cuda", enabled=False):  # spectrogram in fp32
            spec = self.fe(x.cpu().numpy())
            spec = self.torch.as_tensor(np.asarray(spec), dtype=self.torch.float32)
        return self.m.model(spec.cuda())[0]  # ConvNeXt last_hidden_state


class BirdCODE(_ConvNeXtPrototypeEncoder):
    name = "birdcode"

    def __init__(self) -> None:
        super().__init__()
        from sound_event_detection.models import FrameDetector

        self.m = FrameDetector.from_hf_hub(BIRDCODE_REPO).eval().cuda()
        self.enc = self.m.encoder
        self.provenance = {
            "source": f"huggingface {BIRDCODE_REPO}",
            "layer": "encoder GAP",
        }

    def _features(self, x):
        with self.torch.autocast(
            "cuda", enabled=False
        ):  # fp16 mel overflows on loud audio
            mel = self.enc.mel_transform(x.float())
        return self.enc.backbone(mel).last_hidden_state


@register("audioprotopnet-20")
def make_apn(models_dir: Path) -> AudioProtoPNet:
    _hf_cache(models_dir)
    return AudioProtoPNet()


@register("birdcode")
def make_birdcode(models_dir: Path) -> BirdCODE:
    _hf_cache(models_dir)
    return BirdCODE()
