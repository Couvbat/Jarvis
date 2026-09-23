"""Tests for interrupting Jarvis by speaking (speech/barge_in.py)."""

import asyncio

import numpy as np
import pytest

from audio_handler import AudioHandler
from speech.barge_in import (
    CALIBRATION_FRAMES,
    MIN_ENERGY,
    BargeInListener,
    frame_energy,
)
from tests._stubs import FakeInputStream, FakeVad


def tone(samples=320, amplitude=5000):
    """A loud-ish frame."""
    return np.full((samples, 1), amplitude, dtype=np.int16)


def quiet(samples=320, amplitude=50):
    return np.full((samples, 1), amplitude, dtype=np.int16)


@pytest.fixture
def handler(settings):
    settings.chunk_size = 320
    return AudioHandler()


class TestEnergy:
    def test_silence_has_no_energy(self):
        assert frame_energy(np.zeros(320, dtype=np.int16)) == 0.0

    def test_a_louder_frame_has_more_energy(self):
        assert frame_energy(tone(amplitude=8000)) > frame_energy(tone(amplitude=800))

    def test_an_empty_frame_is_handled(self):
        assert frame_energy(np.array([], dtype=np.int16)) == 0.0


class TestDetection:
    async def listen(self, handler, blocks, fake_vad, speech=True,
                     timeout=0.2, **kwargs):
        """Run the listener over scripted audio.

        The timeout is short on purpose: the negative cases are the ones that
        wait it out, and there is nothing to wait for - the stub delivers
        every frame immediately.
        """
        FakeInputStream.scripted_blocks = list(blocks)
        FakeVad.speech_decider = staticmethod(lambda buf: speech)
        listener = BargeInListener(handler, **kwargs)
        await listener.start()
        try:
            await asyncio.wait_for(listener.detected.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return listener, await listener.stop()

    async def test_sustained_speech_is_detected(self, handler, fake_sd, fake_vad):
        blocks = [quiet()] * CALIBRATION_FRAMES + [tone()] * 20
        listener, captured = await self.listen(
            handler, blocks, fake_vad, min_speech_frames=5
        )
        assert listener.detected.is_set()
        assert captured is not None and len(captured) > 0

    async def test_a_brief_noise_is_not_an_interruption(self, handler, fake_sd, fake_vad):
        """A door closing or a key press should not cut Jarvis off."""
        decisions = iter([True] * 2 + [False] * 500)
        FakeInputStream.scripted_blocks = (
            [quiet()] * CALIBRATION_FRAMES + [tone()] * 2 + [quiet()] * 40
        )
        FakeVad.speech_decider = staticmethod(lambda buf: next(decisions))
        listener = BargeInListener(handler, min_speech_frames=10)
        await listener.start()
        await asyncio.sleep(0.1)
        captured = await listener.stop()
        assert listener.detected.is_set() is False
        assert captured is None

    async def test_quiet_speech_below_the_floor_is_ignored(self, handler, fake_sd, fake_vad):
        """The floor is measured on Jarvis's own voice coming back, so echo
        has to be quieter than the person to be ignored."""
        blocks = [tone(amplitude=5000)] * CALIBRATION_FRAMES + [tone(amplitude=5000)] * 40
        listener, captured = await self.listen(
            handler, blocks, fake_vad, min_speech_frames=5
        )
        assert listener.detected.is_set() is False

    async def test_speech_well_above_the_floor_gets_through(self, handler, fake_sd, fake_vad):
        blocks = [tone(amplitude=1000)] * CALIBRATION_FRAMES + [tone(amplitude=20000)] * 40
        listener, _ = await self.listen(handler, blocks, fake_vad, min_speech_frames=5)
        assert listener.detected.is_set() is True

    async def test_the_vad_must_agree_it_is_speech(self, handler, fake_sd, fake_vad):
        blocks = [quiet()] * CALIBRATION_FRAMES + [tone()] * 40
        listener, _ = await self.listen(
            handler, blocks, fake_vad, speech=False, min_speech_frames=5
        )
        assert listener.detected.is_set() is False

    async def test_the_run_must_be_unbroken(self, handler, fake_sd, fake_vad):
        decisions = iter(([True] * 3 + [False]) * 50)
        FakeInputStream.scripted_blocks = [quiet()] * CALIBRATION_FRAMES + [tone()] * 60
        FakeVad.speech_decider = staticmethod(lambda buf: next(decisions))
        listener = BargeInListener(handler, min_speech_frames=6)
        await listener.start()
        await asyncio.sleep(0.15)
        assert listener.detected.is_set() is False
        await listener.stop()

    async def test_a_silent_room_never_calibrates_to_zero(self, handler, fake_sd, fake_vad):
        """Otherwise any sound at all would clear the bar."""
        blocks = [np.zeros((320, 1), dtype=np.int16)] * CALIBRATION_FRAMES + [
            tone(amplitude=int(MIN_ENERGY))
        ] * 40
        listener, _ = await self.listen(handler, blocks, fake_vad, min_speech_frames=5)
        assert listener.detected.is_set() is False


class TestCapturedAudio:
    async def test_the_words_that_interrupted_are_kept(self, handler, fake_sd, fake_vad):
        """Losing the start of a sentence would make interrupting worse than
        waiting for Jarvis to finish."""
        FakeInputStream.scripted_blocks = (
            [quiet()] * CALIBRATION_FRAMES + [tone()] * 20
        )
        FakeVad.speech_decider = staticmethod(lambda buf: True)
        listener = BargeInListener(handler, min_speech_frames=5)
        await listener.start()
        await asyncio.wait_for(listener.detected.wait(), timeout=1.0)
        captured = await listener.stop()
        assert len(captured) == 5 * 320

    async def test_nothing_is_captured_without_a_detection(self, handler, fake_sd, fake_vad):
        FakeInputStream.scripted_blocks = [quiet()] * 40
        FakeVad.speech_decider = staticmethod(lambda buf: False)
        listener = BargeInListener(handler, min_speech_frames=5)
        await listener.start()
        await asyncio.sleep(0.05)
        assert await listener.stop() is None


class TestLifecycle:
    async def test_stopping_without_starting_is_safe(self, handler):
        assert await BargeInListener(handler).stop() is None

    async def test_starting_twice_is_safe(self, handler, fake_sd, fake_vad):
        listener = BargeInListener(handler)
        await listener.start()
        await listener.start()
        await listener.stop()

    async def test_a_refused_device_costs_barge_in_not_the_session(
        self, handler, fake_sd, monkeypatch
    ):
        """A second stream on the same device can simply be refused."""
        class Refused:
            def __init__(self, **kwargs):
                raise OSError("device busy")

        monkeypatch.setattr(fake_sd, "InputStream", Refused)
        listener = BargeInListener(handler)
        await listener.start()
        await asyncio.sleep(0.05)
        assert await listener.stop() is None
        assert listener.detected.is_set() is False

    async def test_an_unsupported_rate_disables_it(self, settings, fake_sd):
        settings.sample_rate = 44100
        listener = BargeInListener(AudioHandler())
        await listener.start()
        await asyncio.sleep(0.05)
        assert await listener.stop() is None


class TestRecordingWithAPrefix:
    def test_the_prefix_is_prepended(self, handler, fake_sd):
        prefix = np.full(640, 1234, dtype=np.int16)
        audio = handler.record_until_silence(max_duration=0.2, prefix=prefix)
        assert len(audio) > 640
        assert int(audio[0]) == 1234

    def test_the_result_is_still_flat_mono(self, handler, fake_sd):
        prefix = np.full((640, 1), 1234, dtype=np.int16)
        assert handler.record_until_silence(max_duration=0.2, prefix=prefix).ndim == 1

    def test_no_prefix_changes_nothing(self, handler, fake_sd):
        plain = handler.record_until_silence(max_duration=0.2)
        with_none = handler.record_until_silence(max_duration=0.2, prefix=None)
        assert len(plain) == len(with_none)

    def test_an_empty_prefix_is_ignored(self, handler, fake_sd):
        empty = np.array([], dtype=np.int16)
        audio = handler.record_until_silence(max_duration=0.2, prefix=empty)
        assert len(audio) > 0

    def test_a_prefix_survives_a_zero_length_recording(self, handler, fake_sd):
        prefix = np.full(640, 1234, dtype=np.int16)
        audio = handler.record_until_silence(max_duration=0.0, prefix=prefix)
        assert len(audio) == 640
