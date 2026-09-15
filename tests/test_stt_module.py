"""Tests for speech-to-text (stt_module.py)."""

import numpy as np
import pytest

from stt_module import STTModule
from tests._stubs import FakeWhisperModel


class TestInitialization:
    def test_reads_settings(self, settings):
        settings.whisper_model = "small"
        settings.whisper_device = "cuda"
        settings.whisper_compute_type = "float16"
        module = STTModule()
        assert (module.model_name, module.device, module.compute_type) == (
            "small", "cuda", "float16",
        )

    def test_model_is_lazy(self):
        assert STTModule().model is None
        assert FakeWhisperModel.instances == []

    def test_initialize_loads_the_model_with_settings(self, settings):
        settings.whisper_model = "tiny"
        module = STTModule()
        module.initialize()
        loaded = FakeWhisperModel.instances[0]
        assert loaded.model_size_or_path == "tiny"
        assert loaded.device == "cpu"
        assert loaded.compute_type == "int8"

    def test_initialize_is_idempotent(self):
        module = STTModule()
        module.initialize()
        module.initialize()
        assert len(FakeWhisperModel.instances) == 1

    def test_load_failure_propagates(self):
        FakeWhisperModel.load_error = RuntimeError("no such model")
        with pytest.raises(RuntimeError, match="no such model"):
            STTModule().initialize()


class TestTranscribe:
    @pytest.fixture
    def module(self):
        instance = STTModule()
        instance.initialize()
        return instance

    def test_requires_initialization(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            STTModule().transcribe(np.zeros(16000, dtype=np.int16))

    def test_returns_every_segment(self, module):
        FakeWhisperModel.scripted_text = "bonjour tout le monde"
        result = module.transcribe(np.zeros(16000, dtype=np.int16))
        assert result == "bonjour tout le monde"

    def test_int16_is_normalised_to_float32(self, module):
        audio = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        module.transcribe(audio)
        passed = FakeWhisperModel.instances[0].transcribe_calls[0]["audio"]
        assert passed.dtype == np.float32
        assert passed.max() <= 1.0 and passed.min() >= -1.0
        assert passed[1] == pytest.approx(0.5, abs=1e-4)

    def test_float_input_is_passed_through(self, module):
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        module.transcribe(audio)
        passed = FakeWhisperModel.instances[0].transcribe_calls[0]["audio"]
        assert passed[1] == pytest.approx(0.5)

    def test_vad_filter_is_enabled(self, module):
        module.transcribe(np.zeros(16000, dtype=np.int16))
        assert FakeWhisperModel.instances[0].transcribe_calls[0]["vad_filter"] is True

    @pytest.mark.parametrize("configured", ["auto", "AUTO", " auto ", ""])
    def test_automatic_detection_is_passed_as_none(self, settings, configured):
        """The README spells it "auto"; faster-whisper wants None, and the
        literal string is not a valid code."""
        settings.whisper_language = configured
        module = STTModule()
        module.initialize()
        module.transcribe(np.zeros(16000, dtype=np.int16))
        assert FakeWhisperModel.instances[0].transcribe_calls[0]["language"] is None

    def test_configured_language_is_forwarded(self, settings):
        settings.whisper_language = "fr"
        module = STTModule()
        module.initialize()
        module.transcribe(np.zeros(16000, dtype=np.int16))
        assert FakeWhisperModel.instances[0].transcribe_calls[0]["language"] == "fr"

    def test_empty_transcription_yields_empty_string(self, module):
        FakeWhisperModel.scripted_text = ""
        assert module.transcribe(np.zeros(16000, dtype=np.int16)) == ""

    def test_engine_error_propagates(self, module):
        FakeWhisperModel.transcribe_error = RuntimeError("decode failed")
        with pytest.raises(RuntimeError, match="decode failed"):
            module.transcribe(np.zeros(16000, dtype=np.int16))

    def test_empty_audio_does_not_crash(self, module):
        assert isinstance(module.transcribe(np.array([], dtype=np.int16)), str)


class TestLanguageSwitching:
    def test_set_language_changes_the_next_transcription(self):
        module = STTModule()
        module.initialize()
        module.set_language("fr")
        module.transcribe(np.zeros(16000, dtype=np.int16))
        assert FakeWhisperModel.instances[0].transcribe_calls[0]["language"] == "fr"

    def test_set_language_does_not_reload_the_model(self):
        module = STTModule()
        module.initialize()
        module.set_language("fr")
        assert len(FakeWhisperModel.instances) == 1

    def test_the_detected_language_is_recorded(self):
        module = STTModule()
        module.initialize()
        module.transcribe(np.zeros(16000, dtype=np.int16))
        assert module.detected_language == "en"

    def test_language_is_exposed_as_an_attribute(self):
        module = STTModule()
        module.set_language("en")
        assert module.language == "en"


class TestTranscribeFile:
    def test_requires_initialization(self, tmp_path):
        with pytest.raises(RuntimeError, match="not initialized"):
            STTModule().transcribe_file(str(tmp_path / "a.wav"))

    def test_passes_the_path_through(self, tmp_path):
        module = STTModule()
        module.initialize()
        FakeWhisperModel.scripted_text = "from file"
        assert module.transcribe_file(str(tmp_path / "a.wav")).split() == [
            "from", "file",
        ]
        assert FakeWhisperModel.instances[0].transcribe_calls[0]["audio"] == str(
            tmp_path / "a.wav"
        )


def test_auto_language_is_translated_to_none(settings):
    """The README documents ``auto`` for language auto-detection.

    faster-whisper wants ``language=None`` for that; the literal string
    ``"auto"`` is not a valid code and the model rejects it.
    """
    settings.whisper_language = "auto"
    module = STTModule()
    module.initialize()
    module.transcribe(np.zeros(16000, dtype=np.int16))
    assert FakeWhisperModel.instances[0].transcribe_calls[0]["language"] is None


def test_segments_are_joined_with_single_spaces():
    """faster-whisper segments already start with a space; ``" ".join`` doubles it.

    The doubled spacing reaches the LLM prompt and the TUI transcript.
    """
    module = STTModule()
    module.initialize()
    FakeWhisperModel.scripted_text = "bonjour tout le monde"
    assert module.transcribe(np.zeros(16000, dtype=np.int16)) == "bonjour tout le monde"


def test_detected_language_is_reported_to_the_caller():
    """``info.language`` is logged and dropped, so nothing can act on a
    mid-conversation language switch by the speaker."""
    module = STTModule()
    module.initialize()
    result = module.transcribe_with_info(np.zeros(16000, dtype=np.int16))
    assert result["language"] == "en"
