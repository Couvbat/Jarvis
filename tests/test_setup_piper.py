"""Tests for the Piper installer (setup_piper.py)."""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

import setup_piper
from setup_piper import (
    BINARIES,
    DEFAULT_VOICE,
    VOICES,
    Artefact,
    VerificationError,
    extract_safely,
    fetch_verified,
    sha256_of,
)
from setup_piper import (
    setup_piper as install,
)

BINARY_PAYLOAD = b"#!/bin/sh\necho piper 1.2.0\n"


def digest_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def piper_tarball(path: Path, payload: bytes = BINARY_PAYLOAD) -> bytes:
    """A tarball shaped like the real Piper release."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("piper/piper")
        info.size = len(payload)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def evil_tarball(path: Path) -> bytes:
    """A tarball that tries to write outside the target directory."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        payload = b"pwned"
        info = tarfile.TarInfo("../escaped.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


@pytest.fixture
def linux_x86(monkeypatch):
    monkeypatch.setattr(setup_piper.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_piper.platform, "machine", lambda: "x86_64")


@pytest.fixture
def fake_network(monkeypatch, tmp_path):
    """Serve scripted bytes for each URL, and record what was asked for."""
    served = {}
    requested = []

    def fake_download(url, dest):
        requested.append(url)
        if url not in served:
            raise OSError(f"nothing served at {url}")
        Path(dest).write_bytes(served[url])

    monkeypatch.setattr(setup_piper, "download_file", fake_download)
    fake_download.served = served
    fake_download.requested = requested
    return fake_download


def serve_real_layout(network, tmp_path, corrupt=None):
    """Serve a binary and both voices, with real digests pinned in-place."""
    archive_bytes = piper_tarball(tmp_path / "scratch" / "piper.tar.gz")
    binary = BINARIES[("Linux", "x86_64")]
    network.served[binary.url] = archive_bytes

    voice = VOICES[DEFAULT_VOICE]
    network.served[voice.model.url] = b"fake-model"
    network.served[voice.config.url] = b"fake-config"

    patched = {
        binary.url: digest_of(archive_bytes),
        voice.model.url: digest_of(b"fake-model"),
        voice.config.url: digest_of(b"fake-config"),
    }
    if corrupt:
        network.served[corrupt] = b"something else entirely"
    return patched


@pytest.fixture
def pinned(monkeypatch):
    """Replace the pinned digests with ones matching the served bytes."""
    def apply(mapping):
        monkeypatch.setitem(
            BINARIES, ("Linux", "x86_64"),
            Artefact(BINARIES[("Linux", "x86_64")].url,
                     mapping[BINARIES[("Linux", "x86_64")].url]),
        )
        voice = VOICES[DEFAULT_VOICE]
        monkeypatch.setitem(VOICES, DEFAULT_VOICE, setup_piper.Voice(
            model=Artefact(voice.model.url, mapping[voice.model.url]),
            config=Artefact(voice.config.url, mapping[voice.config.url]),
            language=voice.language,
        ))
    return apply


class TestDigests:
    def test_hashing_a_file(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"hello")
        assert sha256_of(path) == digest_of(b"hello")

    def test_large_files_are_read_in_pieces(self, tmp_path):
        path = tmp_path / "big.bin"
        payload = b"x" * (setup_piper.CHUNK * 2 + 17)
        path.write_bytes(payload)
        assert sha256_of(path) == digest_of(payload)

    def test_every_pinned_artefact_has_a_full_digest(self):
        artefacts = list(BINARIES.values())
        for voice in VOICES.values():
            artefacts.extend([voice.model, voice.config])
        for artefact in artefacts:
            assert len(artefact.sha256) == 64
            assert artefact.sha256 == artefact.sha256.lower()


class TestVerification:
    def test_a_matching_download_is_kept(self, tmp_path, fake_network):
        artefact = Artefact("https://example.test/f", digest_of(b"payload"))
        fake_network.served[artefact.url] = b"payload"
        assert fetch_verified(artefact, tmp_path / "f").read_bytes() == b"payload"

    def test_a_mismatched_download_raises(self, tmp_path, fake_network):
        artefact = Artefact("https://example.test/f", digest_of(b"expected"))
        fake_network.served[artefact.url] = b"something else"
        with pytest.raises(VerificationError, match="does not match"):
            fetch_verified(artefact, tmp_path / "f")

    def test_a_mismatched_download_is_deleted(self, tmp_path, fake_network):
        """Leaving it invites a later run to pick it up."""
        artefact = Artefact("https://example.test/f", digest_of(b"expected"))
        fake_network.served[artefact.url] = b"something else"
        with pytest.raises(VerificationError):
            fetch_verified(artefact, tmp_path / "f")
        assert not (tmp_path / "f").exists()

    def test_an_already_verified_file_is_not_downloaded_again(self, tmp_path, fake_network):
        artefact = Artefact("https://example.test/f", digest_of(b"payload"))
        (tmp_path / "f").write_bytes(b"payload")
        fetch_verified(artefact, tmp_path / "f")
        assert fake_network.requested == []

    def test_a_stale_file_is_replaced(self, tmp_path, fake_network):
        artefact = Artefact("https://example.test/f", digest_of(b"payload"))
        fake_network.served[artefact.url] = b"payload"
        (tmp_path / "f").write_bytes(b"an older, different file")
        assert fetch_verified(artefact, tmp_path / "f").read_bytes() == b"payload"

    def test_the_error_names_the_url(self, tmp_path, fake_network):
        artefact = Artefact("https://example.test/f", digest_of(b"expected"))
        fake_network.served[artefact.url] = b"other"
        with pytest.raises(VerificationError, match="example.test"):
            fetch_verified(artefact, tmp_path / "f")


class TestExtraction:
    def test_a_normal_archive_extracts(self, tmp_path):
        archive = tmp_path / "piper.tar.gz"
        piper_tarball(archive)
        extract_safely(archive, tmp_path / "out")
        assert (tmp_path / "out" / "piper" / "piper").read_bytes() == BINARY_PAYLOAD

    def test_an_archive_escaping_the_target_is_refused(self, tmp_path):
        """A crafted tarball could otherwise write anywhere the user can -
        and this one is then marked executable."""
        archive = tmp_path / "evil.tar.gz"
        evil_tarball(archive)
        target = tmp_path / "out"
        # tarfile's data filter raises a FilterError; the manual check raises
        # VerificationError. Either is a refusal.
        with pytest.raises((VerificationError, tarfile.TarError)):
            extract_safely(archive, target)
        assert not (tmp_path / "escaped.txt").exists()

    def test_the_manual_check_catches_it_too(self, tmp_path, monkeypatch):
        """Interpreters without tarfile's data filter must still be safe."""
        real_extractall = tarfile.TarFile.extractall

        def without_filter(self, path=".", members=None, **kwargs):
            if "filter" in kwargs:
                raise TypeError("extractall() got an unexpected keyword 'filter'")
            return real_extractall(self, path, members)

        monkeypatch.setattr(tarfile.TarFile, "extractall", without_filter)

        archive = tmp_path / "evil.tar.gz"
        evil_tarball(archive)
        with pytest.raises(VerificationError, match="outside"):
            extract_safely(archive, tmp_path / "out")
        assert not (tmp_path / "escaped.txt").exists()

    def test_the_manual_check_allows_a_normal_archive(self, tmp_path, monkeypatch):
        real_extractall = tarfile.TarFile.extractall

        def without_filter(self, path=".", members=None, **kwargs):
            if "filter" in kwargs:
                raise TypeError("no filter here")
            return real_extractall(self, path, members)

        monkeypatch.setattr(tarfile.TarFile, "extractall", without_filter)
        archive = tmp_path / "piper.tar.gz"
        piper_tarball(archive)
        extract_safely(archive, tmp_path / "out")
        assert (tmp_path / "out" / "piper" / "piper").exists()


class TestInstallation:
    def test_a_full_install(self, tmp_path, monkeypatch, linux_x86, fake_network, pinned):
        monkeypatch.chdir(tmp_path)
        pinned(serve_real_layout(fake_network, tmp_path))
        assert install(piper_dir=tmp_path / "piper") is True

        binary = tmp_path / "piper" / "piper" / "piper"
        assert binary.exists() and binary.stat().st_mode & 0o111
        assert (tmp_path / "piper" / "models" / f"{DEFAULT_VOICE}.onnx").exists()
        assert (tmp_path / "piper" / "models" / f"{DEFAULT_VOICE}.onnx.json").exists()

    def test_the_layout_matches_what_the_tts_module_probes(
        self, tmp_path, monkeypatch, linux_x86, fake_network, pinned
    ):
        monkeypatch.chdir(tmp_path)
        pinned(serve_real_layout(fake_network, tmp_path))
        install(piper_dir=tmp_path / "piper")

        # The probe list in tts_module contains this exact path (BUG-15).
        assert (tmp_path / "piper" / "piper" / "piper").exists()
        assert (tmp_path / "piper" / "models").is_dir()

    def test_a_corrupt_binary_stops_the_install(
        self, tmp_path, monkeypatch, linux_x86, fake_network, pinned
    ):
        monkeypatch.chdir(tmp_path)
        mapping = serve_real_layout(fake_network, tmp_path)
        pinned(mapping)
        fake_network.served[BINARIES[("Linux", "x86_64")].url] = b"not the archive"

        assert install(piper_dir=tmp_path / "piper") is False
        assert not (tmp_path / "piper" / "piper" / "piper").exists()

    def test_a_corrupt_voice_stops_the_install(
        self, tmp_path, monkeypatch, linux_x86, fake_network, pinned
    ):
        monkeypatch.chdir(tmp_path)
        pinned(serve_real_layout(fake_network, tmp_path))
        fake_network.served[VOICES[DEFAULT_VOICE].model.url] = b"not the model"
        assert install(piper_dir=tmp_path / "piper") is False


class TestVoices:
    def test_a_french_voice_is_available(self):
        """WHISPER_LANGUAGE defaults to fr, so a French install needs one."""
        assert any(v.language == "French" for v in VOICES.values())

    def test_an_unknown_voice_is_refused(self, tmp_path, linux_x86):
        assert install(voice="xx_XX-nobody-medium", piper_dir=tmp_path) is False

    def test_each_voice_pins_both_files(self):
        for name, voice in VOICES.items():
            assert voice.model.url.endswith(".onnx"), name
            assert voice.config.url.endswith(".onnx.json"), name


class TestPlatformSupport:
    def test_both_linux_architectures_are_published(self):
        assert ("Linux", "x86_64") in BINARIES
        assert ("Linux", "aarch64") in BINARIES

    def test_an_unsupported_platform_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_piper.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(setup_piper.platform, "machine", lambda: "arm64")
        assert install(piper_dir=tmp_path) is False

    def test_an_unsupported_architecture_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_piper.platform, "system", lambda: "Linux")
        monkeypatch.setattr(setup_piper.platform, "machine", lambda: "riscv64")
        assert install(piper_dir=tmp_path) is False
