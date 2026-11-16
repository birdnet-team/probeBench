from __future__ import annotations

import zipfile

import pytest

from probebench import models
from probebench.models import birdnet_zenodo as bz


def test_builtin_specs_are_registered():
    assert {"mel", "birdnet-v3.0", "birdnet-v2.4"} <= set(models.available())
    with pytest.raises(ValueError, match="unknown model"):
        models.from_spec("no-such-model")


def test_cached_model_file_needs_no_network(tmp_path, monkeypatch):
    def offline(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(bz, "zenodo", offline)
    f = tmp_path / f"zenodo_{bz.V30['record']}" / bz.V30["file"]
    f.parent.mkdir()
    f.write_bytes(b"onnx")
    assert bz.model_file(bz.V30, tmp_path) == f


def test_zip_member_is_extracted_and_zip_removed(tmp_path, monkeypatch):
    folder = tmp_path / f"zenodo_{bz.V24['record']}"
    folder.mkdir()
    archive = folder / bz.V24["file"]
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(bz.V24["member"], b"tflite")
        z.writestr("labels/en_us.txt", b"x")
    monkeypatch.setattr(bz, "_remote", lambda record, name: None)
    monkeypatch.setattr(bz, "fetch", lambda remote, folder: archive)
    out = bz.model_file(bz.V24, tmp_path)
    assert out.read_bytes() == b"tflite"
    assert not archive.exists()
