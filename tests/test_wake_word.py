"""Tests for hands-free activation (speech/wake.py, AudioHandler.wait_for_wake).

Two things decide whether a wake word is pleasant or infuriating, and both are
tested here rather than assumed: the command said in the same breath as the
phrase must survive, and a detector that is missing or broken must cost the
feature, not the assistant.
"""

import numpy as np
import pytest

from jarvis.audio_handler import AudioHandler
from jarvis.speech.wake import (
    FRAME_SAMPLES,
    MODEL_SAMPLE_RATE,
    WakeWord,
    WakeWordUnavailable,
)
from tests._stubs import FakeInputStream, FakeWakeModel


def blocks(*values, size=320):
    """Scripted capture blocks, one per value, as (frames, 1) int16."""
    return [np.full((size, 1), value, dtype=np.int16) for value in values]


class ScriptedDetector:
    """A detector that fires on a chosen call, for testing the capture loop."""

    def __init__(self, fire_on=1):
        self.fire_on = fire_on
        self.calls = 0
        self.resets = 0
        self.fed = []

    def feed(self, audio):
        self.calls += 1
        self.fed.append(np.asarray(audio).copy())
        return self.calls == self.fire_on

    def reset(self):
        self.resets += 1


class TestConstruction:
    def test_a_model_is_loaded_for_the_configured_phrase(self, fake_ollama):
        WakeWord(model="hey_jarvis")
        assert FakeWakeModel.instances[-1].wakeword_models == ["hey_jarvis"]

    def test_a_missing_package_is_a_clear_refusal(self, monkeypatch):
        """Not a traceback about an import: the fix is two commands, and the
        message has to carry both."""
        import sys

        monkeypatch.setitem(sys.modules, "openwakeword.model", None)
        with pytest.raises(WakeWordUnavailable) as raised:
            WakeWord()
        assert "pip install" in str(raised.value)

    def test_missing_models_are_reported_separately(self):
        """Installing the package and downloading the models are two steps,
        and only one of them is `pip install`."""
        import openwakeword

        openwakeword.load_error = OSError("no such file: hey_jarvis_v0.1.tflite")
        with pytest.raises(WakeWordUnavailable) as raised:
            WakeWord()
        assert "download_models" in str(raised.value)

    def test_a_wrong_sample_rate_is_refused_up_front(self):
        """The models only mean anything at 16 kHz. Failing at construction
        beats scoring noise at every frame for the rest of the session."""
        with pytest.raises(WakeWordUnavailable) as raised:
            WakeWord(sample_rate=44100)
        assert "16000" in str(raised.value)

    def test_it_describes_itself_for_the_log(self):
        assert "hey_jarvis" in WakeWord().describe()


class TestDetection:
    def detector(self, *scores, threshold=0.5):
        FakeWakeModel.scripted_scores = list(scores)
        return WakeWord(threshold=threshold)

    def test_a_score_above_the_threshold_fires(self):
        wake = self.detector(0.9)
        assert wake.feed(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is True

    def test_a_score_below_the_threshold_does_not(self):
        wake = self.detector(0.2)
        assert wake.feed(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False

    def test_the_threshold_is_inclusive(self):
        wake = self.detector(0.5, threshold=0.5)
        assert wake.feed(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is True

    def test_the_model_is_fed_the_frame_size_it_wants(self, fake_ollama):
        """openWakeWord buffers anything else internally, which only adds
        latency; 1280 samples is 80 ms at 16 kHz."""
        wake = self.detector(0.0, 0.0, 0.0)
        wake.feed(np.zeros(FRAME_SAMPLES * 3, dtype=np.int16))
        assert FakeWakeModel.instances[-1].frame_sizes == [FRAME_SAMPLES] * 3

    def test_a_partial_frame_is_held_for_the_next_block(self):
        """Capture blocks are not frame-sized, so a detection must not be lost
        at a block boundary."""
        wake = self.detector(0.9)
        half = FRAME_SAMPLES // 2

        assert wake.feed(np.zeros(half, dtype=np.int16)) is False
        assert FakeWakeModel.instances[-1].frame_sizes == []
        assert wake.feed(np.zeros(half, dtype=np.int16)) is True

    def test_a_detection_anywhere_in_a_block_is_reported(self):
        wake = self.detector(0.0, 0.0, 0.9)
        assert wake.feed(np.zeros(FRAME_SAMPLES * 3, dtype=np.int16)) is True

    def test_empty_audio_is_not_a_detection(self):
        wake = self.detector(0.9)
        assert wake.feed(np.array([], dtype=np.int16)) is False
        assert wake.feed(None) is False

    def test_a_failing_detector_does_not_kill_the_loop(self, monkeypatch):
        """A model that raises would otherwise take the assistant down with
        it; silence is the safe reading of a frame nothing could score."""
        wake = self.detector(0.9)
        monkeypatch.setattr(
            wake._detector, "predict",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("onnx died")),
        )
        assert wake.feed(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False

    def test_an_empty_score_map_is_not_a_detection(self, monkeypatch):
        wake = self.detector(0.9)
        monkeypatch.setattr(wake._detector, "predict", lambda *a, **k: {})
        assert wake.feed(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False

    def test_reset_clears_both_sides(self):
        """The half frame held here and the model's own streaming state: a
        leftover from the last session would score against the next one."""
        wake = self.detector(0.9)
        wake.feed(np.zeros(FRAME_SAMPLES // 2, dtype=np.int16))
        wake.reset()

        assert wake._detector.resets == 1
        assert wake.feed(np.zeros(FRAME_SAMPLES // 2, dtype=np.int16)) is False

    def test_a_detector_without_reset_is_tolerated(self):
        """`detector` is an injection point; not everything passed in is an
        openWakeWord model."""
        class Bare:
            def predict(self, frame):
                return {"x": 0.0}

        WakeWord(detector=Bare()).reset()


class TestWaitingForTheWakeWord:
    def handler(self, settings):
        settings.sample_rate = MODEL_SAMPLE_RATE
        settings.chunk_size = 320
        return AudioHandler()

    def test_nothing_is_returned_before_the_phrase(self, fake_sd, settings):
        """Listening is not recording: the room before the wake word is not
        the user's command and must not reach the transcriber."""
        FakeInputStream.scripted_blocks = blocks(1, 2, 3)
        detector = ScriptedDetector(fire_on=10_000)  # never fires

        stops = iter([False, True])
        audio = self.handler(settings).wait_for_wake(
            detector, should_stop=lambda: next(stops)
        )
        assert len(audio) == 0

    def test_the_audio_after_the_phrase_is_returned(self, fake_sd, settings):
        """"hey Jarvis quelle heure est-il" is one breath. Dropping what comes
        after the phrase makes the user say the command twice."""
        FakeInputStream.scripted_blocks = blocks(*range(1, 200))
        detector = ScriptedDetector(fire_on=1)

        audio = self.handler(settings).wait_for_wake(detector, tail=0.1)
        assert len(audio) >= int(0.1 * MODEL_SAMPLE_RATE)

    def test_the_tail_length_is_honoured(self, fake_sd, settings):
        FakeInputStream.scripted_blocks = blocks(*range(1, 400))
        detector = ScriptedDetector(fire_on=1)

        handler = self.handler(settings)
        short = handler.wait_for_wake(detector, tail=0.05)
        long = handler.wait_for_wake(ScriptedDetector(fire_on=1), tail=0.3)
        assert len(long) > len(short)

    def test_the_detector_is_reset_before_listening(self, fake_sd, settings):
        """Otherwise the words that woke the last session can wake this one."""
        detector = ScriptedDetector(fire_on=1)
        self.handler(settings).wait_for_wake(detector, tail=0.05)
        assert detector.resets == 1

    def test_the_detector_is_fed_mono(self, fake_sd, settings):
        """The stream can deliver (frames, channels); the model only takes a
        flat waveform."""
        settings.channels = 2
        FakeInputStream.scripted_blocks = [
            np.full((320, 2), 100, dtype=np.int16) for _ in range(5)
        ]
        detector = ScriptedDetector(fire_on=1)

        self.handler(settings).wait_for_wake(detector, tail=0.01)
        assert detector.fed[0].ndim == 1

    def test_stopping_is_checked_between_blocks(self, fake_sd, settings):
        """A voice loop waits forever; a shutdown has to be able to say when."""
        detector = ScriptedDetector(fire_on=10_000)
        audio = self.handler(settings).wait_for_wake(
            detector, should_stop=lambda: True
        )
        assert len(audio) == 0

    def test_a_stream_failure_is_raised_not_swallowed(self, fake_sd, settings):
        """A microphone that cannot be opened is the caller's problem to
        report; returning empty audio would look like silence forever."""
        class Broken:
            def __init__(self, **kwargs):
                raise OSError("device unavailable")

        fake_sd.InputStream = Broken
        try:
            with pytest.raises(OSError):
                self.handler(settings).wait_for_wake(ScriptedDetector())
        finally:
            fake_sd.InputStream = FakeInputStream
