"""Tests for the Piper installer (setup_piper.py)."""

import io
import tarfile
from pathlib import Path

import pytest

import setup_piper


@pytest.fixture
def installer(tmp_path, monkeypatch):
    """Run the installer in a temp dir with downloads and extraction faked."""
    monkeypatch.chdir(tmp_path)
    downloads = []

    def fake_download(url, dest):
        downloads.append((url, Path(dest)))
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.name.endswith(".tar.gz"):
            _write_piper_tarball(dest)
        else:
            dest.write_bytes(b"fake-model")

    monkeypatch.setattr(setup_piper, "download_file", fake_download)
    return downloads


def _write_piper_tarball(path: Path) -> None:
    """Build a tarball with the same layout as the real Piper release."""
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("piper/piper")
        payload = b"#!/bin/sh\necho piper 1.2.0\n"
        info.size = len(payload)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))


class TestPlatformSupport:
    def test_linux_x86_64_is_supported(self, installer, monkeypatch):
        monkeypatch.setattr(setup_piper.platform, "system", lambda: "Linux")
        monkeypatch.setattr(setup_piper.platform, "machine", lambda: "x86_64")
        assert setup_piper.setup_piper() is True
        assert "piper_amd64.tar.gz" in installer[0][0]

    def test_linux_aarch64_is_supported(self, installer, monkeypatch):
        monkeypatch.setattr(setup_piper.platform, "system", lambda: "Linux")
        monkeypatch.setattr(setup_piper.platform, "machine", lambda: "aarch64")
        assert setup_piper.setup_piper() is True
        assert "piper_arm64.tar.gz" in installer[0][0]

    def test_unsupported_architecture_is_refused(self, installer, monkeypatch):
        monkeypatch.setattr(setup_piper.platform, "system", lambda: "Linux")
        monkeypatch.setattr(setup_piper.platform, "machine", lambda: "riscv64")
        assert setup_piper.setup_piper() is False
        assert installer == []

    def test_non_linux_is_refused(self, installer, monkeypatch):
        monkeypatch.setattr(setup_piper.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(setup_piper.platform, "machine", lambda: "arm64")
        assert setup_piper.setup_piper() is False
        assert installer == []


class TestInstallation:
    @pytest.fixture(autouse=True)
    def linux_x86(self, monkeypatch):
        monkeypatch.setattr(setup_piper.platform, "system", lambda: "Linux")
        monkeypatch.setattr(setup_piper.platform, "machine", lambda: "x86_64")

    def test_binary_is_extracted_and_made_executable(self, installer, tmp_path):
        setup_piper.setup_piper()
        binary = tmp_path / "piper" / "piper" / "piper"
        assert binary.exists()
        assert binary.stat().st_mode & 0o111

    def test_voice_model_and_config_are_downloaded(self, installer, tmp_path):
        setup_piper.setup_piper()
        models = tmp_path / "piper" / "models"
        assert (models / "en_US-lessac-medium.onnx").exists()
        assert (models / "en_US-lessac-medium.onnx.json").exists()

    def test_existing_archive_is_not_downloaded_again(self, installer, tmp_path):
        setup_piper.setup_piper()
        first_count = len(installer)
        setup_piper.setup_piper()
        assert len(installer) == first_count

    def test_layout_matches_what_run_sh_probes(self, installer, tmp_path):
        """``run.sh`` checks ``piper/piper/piper`` before running the setup."""
        setup_piper.setup_piper()
        assert (tmp_path / "piper" / "piper" / "piper").exists()


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason="BUG-31: downloads are not integrity-checked")
def test_downloads_are_verified(installer, monkeypatch):
    """Nothing checks a checksum or signature before the archive is extracted
    and marked executable."""
    assert hasattr(setup_piper, "verify_checksum")


@pytest.mark.xfail(strict=True, reason="BUG-32: tar is extracted without a member filter")
def test_extraction_refuses_paths_outside_the_target(installer, tmp_path, monkeypatch):
    """``tar.extractall`` without ``filter='data'`` lets a crafted archive
    write anywhere on disk (CVE-2007-4559 class).  Python 3.14 makes the safe
    filter the default; on 3.11-3.13 it must be passed explicitly."""
    def _write_evil_tarball(path):
        with tarfile.open(path, "w:gz") as tar:
            info = tarfile.TarInfo("../escaped.txt")
            payload = b"pwned"
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    monkeypatch.setattr(setup_piper.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_piper.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        setup_piper, "download_file",
        lambda url, dest: _write_evil_tarball(Path(dest))
        if str(dest).endswith(".tar.gz") else Path(dest).write_bytes(b"x"),
    )
    setup_piper.setup_piper()
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.xfail(
    strict=True, reason="BUG-33: the installer only fetches the English voice"
)
def test_a_french_voice_can_be_installed(installer, monkeypatch):
    """WHISPER_LANGUAGE defaults to ``fr`` but no French Piper voice is ever
    downloaded, so a default install has no voice for the default language."""
    setup_piper.setup_piper(model="fr_FR-siwis-medium")
