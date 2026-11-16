from __future__ import annotations

import tarfile

from probebench.models import perch


def test_cached_saved_model_needs_no_network(tmp_path, monkeypatch):
    def offline(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(perch, "fetch", offline)
    sm = tmp_path / "kaggle_perch_v2_cpu_v1" / "x"
    sm.mkdir(parents=True)
    (sm / "saved_model.pb").write_bytes(b"pb")
    assert perch.saved_model_dir("perch_v2_cpu", tmp_path) == sm


def test_archive_is_extracted_and_removed(tmp_path, monkeypatch):
    folder = tmp_path / "kaggle_perch_v2_v2"
    folder.mkdir()
    src = tmp_path / "saved_model.pb"
    src.write_bytes(b"pb")
    archive = folder / "perch_v2_v2.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        t.add(src, arcname="saved_model.pb")
    monkeypatch.setattr(perch, "fetch", lambda remote, folder: archive)
    assert perch.saved_model_dir("perch_v2", tmp_path) == folder
    assert not archive.exists()
