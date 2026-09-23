"""Tests for text-to-speech (tts_module.py)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import jarvis.tts_module as tts_module
from jarvis.tts_module import TTSModule


class FakeCompleted:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def piper_probe(monkeypatch):
    """Make binary probing deterministic: only ``piper`` on PATH answers."""
    probed = []

    def fake_run(cmd, **kwargs):
        probed.append(cmd)
        if cmd[:2] == ["piper", "--version"]:
            return FakeCompleted(returncode=0)
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(tts_module.subprocess, "run", fake_run)
    return probed


@pytest.fixture
def module(piper_probe):
    return TTSModule()


def write_wav(path: Path, seconds: float = 0.1, sample_rate: int = 22050) -> None:
    samples = np.zeros(int(seconds * sample_rate), dtype=np.int16)
    sf.write(str(path), samples, sample_rate)


class TestBinaryDiscovery:
    def test_finds_the_binary_on_path(self, module):
        assert module.piper_binary == "piper"

    def test_probes_candidates_in_order(self, piper_probe, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd[0])
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(tts_module.subprocess, "run", fake_run)
        TTSModule()
        assert calls[0] == "piper"
        assert "./piper/piper" in calls

    def test_falls_back_to_path_when_nothing_answers(self, monkeypatch):
        monkeypatch.setattr(
            tts_module.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(FileNotFoundError(cmd[0])),
        )
        assert TTSModule().piper_binary == "piper"

    def test_a_timeout_does_not_abort_discovery(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[0] == "piper":
                raise subprocess.TimeoutExpired(cmd, 2)
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(tts_module.subprocess, "run", fake_run)
        assert TTSModule().piper_binary == "piper"

    def test_nonzero_exit_is_not_accepted(self, monkeypatch):
        probed = []

        def fake_run(cmd, **kwargs):
            probed.append(cmd[0])
            return FakeCompleted(returncode=1)

        monkeypatch.setattr(tts_module.subprocess, "run", fake_run)
        TTSModule()
        assert len(probed) == 5  # every candidate was tried


class TestModelDiscovery:
    def test_finds_a_model_in_the_local_directory(self, module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        models = tmp_path / "piper" / "models"
        models.mkdir(parents=True)
        (models / "en_US-lessac-medium.onnx").write_bytes(b"fake")
        assert module._find_model_path() == Path(
            "./piper/models/en_US-lessac-medium.onnx"
        )

    def test_missing_model_falls_back_to_the_bare_filename(
        self, module, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        assert module._find_model_path() == Path("en_US-lessac-medium.onnx")

    def test_initialize_sets_the_model_path(self, module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        module.initialize()
        assert module.model_path is not None

    def test_initialize_reads_the_sample_rate_from_the_voice_config(
        self, module, tmp_path, monkeypatch
    ):
        """Piper ships an ``.onnx.json`` next to each voice; it names the rate."""
        monkeypatch.chdir(tmp_path)
        models = tmp_path / "piper" / "models"
        models.mkdir(parents=True)
        (models / "en_US-lessac-medium.onnx").write_bytes(b"fake")
        (models / "en_US-lessac-medium.onnx.json").write_text(
            json.dumps({"audio": {"sample_rate": 16000}})
        )
        module.initialize()
        assert module.sample_rate == 16000

    def test_a_missing_voice_config_falls_back_to_the_default_rate(
        self, module, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        module.initialize()
        assert module.sample_rate == tts_module.DEFAULT_SAMPLE_RATE

    def test_a_corrupt_voice_config_falls_back_to_the_default_rate(
        self, module, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        models = tmp_path / "piper" / "models"
        models.mkdir(parents=True)
        (models / "en_US-lessac-medium.onnx").write_bytes(b"fake")
        (models / "en_US-lessac-medium.onnx.json").write_text("{not json")
        module.initialize()
        assert module.sample_rate == tts_module.DEFAULT_SAMPLE_RATE

    def test_initialize_survives_a_missing_binary(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            tts_module.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(FileNotFoundError(cmd[0])),
        )
        TTSModule().initialize()  # must not raise


class TestSynthesize:
    @pytest.fixture
    def synth(self, module, tmp_path, monkeypatch):
        """Patch piper to emit a real wav file at the requested output path."""
        monkeypatch.chdir(tmp_path)
        module.initialize()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            output = Path(cmd[cmd.index("--output_file") + 1])
            write_wav(output, seconds=captured.get("seconds", 0.1),
                      sample_rate=captured.get("sample_rate", 22050))
            return FakeCompleted(returncode=0)

        monkeypatch.setattr(tts_module.subprocess, "run", fake_run)
        module._captured = captured
        return module

    def test_empty_text_returns_empty_audio_without_calling_piper(self, synth):
        audio, _ = synth.synthesize("")
        assert audio.size == 0
        assert "cmd" not in synth._captured

    def test_empty_text_still_reports_a_usable_sample_rate(self, synth):
        _, sample_rate = synth.synthesize("")
        assert sample_rate > 0

    def test_whitespace_only_text_returns_empty_audio(self, synth):
        audio, _ = synth.synthesize("   ")
        assert audio.size == 0

    def test_returns_int16_audio(self, synth):
        audio, _ = synth.synthesize("hello")
        assert audio.dtype == np.int16
        assert audio.size > 0

    def test_reports_the_rate_of_the_synthesised_file(self, synth, monkeypatch):
        def emit_16k(cmd, **kwargs):
            write_wav(Path(cmd[cmd.index("--output_file") + 1]), sample_rate=16000)
            return FakeCompleted(returncode=0)

        monkeypatch.setattr(tts_module.subprocess, "run", emit_16k)
        _, sample_rate = synth.synthesize("hello")
        assert sample_rate == 16000
        assert synth.sample_rate == 16000

    def test_passes_the_model_path(self, synth):
        synth.synthesize("hello")
        cmd = synth._captured["cmd"]
        assert "--model" in cmd
        assert str(synth.model_path) == cmd[cmd.index("--model") + 1]

    def test_speaker_id_is_omitted_when_zero(self, synth):
        synth.synthesize("hello")
        assert "--speaker" not in synth._captured["cmd"]

    def test_speaker_id_is_passed_when_set(self, synth):
        synth.speaker_id = 3
        synth.synthesize("hello")
        cmd = synth._captured["cmd"]
        assert cmd[cmd.index("--speaker") + 1] == "3"

    def test_a_timeout_is_set(self, synth):
        synth.synthesize("hello")
        assert synth._captured["kwargs"].get("timeout") is not None

    def test_piper_failure_raises(self, synth, monkeypatch):
        monkeypatch.setattr(
            tts_module.subprocess, "run",
            lambda cmd, **kw: FakeCompleted(returncode=1, stderr=b"model missing"),
        )
        with pytest.raises(RuntimeError, match="Piper synthesis failed"):
            synth.synthesize("hello")

    def test_timeout_raises_a_clear_error(self, synth, monkeypatch):
        def raise_timeout(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 30)

        monkeypatch.setattr(tts_module.subprocess, "run", raise_timeout)
        with pytest.raises(RuntimeError, match="timed out"):
            synth.synthesize("hello")

    def test_temporary_wav_is_removed_on_success(self, synth, tmp_path):
        import tempfile

        before = set(Path(tempfile.gettempdir()).glob("*.wav"))
        synth.synthesize("hello")
        after = set(Path(tempfile.gettempdir()).glob("*.wav"))
        assert after == before


class TestSynthesizeToFile:
    def test_writes_to_the_requested_path(self, module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        module.initialize()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["input"] = kwargs.get("input")
            return FakeCompleted(returncode=0)

        monkeypatch.setattr(tts_module.subprocess, "run", fake_run)
        module.synthesize_to_file("hello", str(tmp_path / "out.wav"))
        assert captured["cmd"][-1] == str(tmp_path / "out.wav")
        assert captured["input"] == b"hello"

    def test_failure_raises(self, module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        module.initialize()
        monkeypatch.setattr(
            tts_module.subprocess, "run", lambda cmd, **kw: FakeCompleted(returncode=1)
        )
        with pytest.raises(RuntimeError):
            module.synthesize_to_file("hello", str(tmp_path / "out.wav"))


def test_local_piper_directory_does_not_break_construction(tmp_path, monkeypatch):
    """``setup_piper.py`` extracts the binary to ``piper/piper/piper``.

    ``./piper/piper`` is then a *directory*, and executing it raises
    ``PermissionError`` - which used to escape the probe and crash
    construction on exactly the layout the project's own installer produces.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "piper" / "piper").mkdir(parents=True)

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] in ("piper", str(Path.home() / ".local/bin/piper"),
                      "/usr/local/bin/piper"):
            raise FileNotFoundError(cmd[0])
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(tts_module.subprocess, "run", fake_run)
    TTSModule()


def test_synthesize_reports_its_sample_rate(tmp_path, monkeypatch, piper_probe):
    """A ``*-low`` voice is 16 kHz; played at an assumed 22050 it would run
    ~37% too fast."""
    monkeypatch.chdir(tmp_path)
    module = TTSModule()
    module.initialize()

    def fake_run(cmd, **kwargs):
        write_wav(Path(cmd[cmd.index("--output_file") + 1]), sample_rate=16000)
        return FakeCompleted(returncode=0)

    monkeypatch.setattr(tts_module.subprocess, "run", fake_run)
    audio, sample_rate = module.synthesize("hello")
    assert sample_rate == 16000


def test_temp_files_are_cleaned_up_when_piper_fails(tmp_path, monkeypatch, piper_probe):
    """A failed synthesis must not leave its .txt and .wav behind."""
    import tempfile

    monkeypatch.chdir(tmp_path)
    module = TTSModule()
    module.initialize()
    monkeypatch.setattr(
        tts_module.subprocess, "run",
        lambda cmd, **kw: FakeCompleted(returncode=1, stderr=b"nope"),
    )
    temp_dir = Path(tempfile.gettempdir())
    before = set(temp_dir.glob("tmp*"))
    with pytest.raises(RuntimeError):
        module.synthesize("hello")
    assert set(temp_dir.glob("tmp*")) == before
