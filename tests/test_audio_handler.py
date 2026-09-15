"""Tests for recording, VAD and playback (audio_handler.py)."""

import numpy as np
import pytest

from audio_handler import AudioHandler
from tests._stubs import FakeInputStream, FakeVad


def block(samples: int, channels: int = 1, value: int = 0) -> np.ndarray:
    return np.full((samples, channels), value, dtype=np.int16)


@pytest.fixture
def handler(settings):
    return AudioHandler()


class TestConstruction:
    def test_reads_settings(self, settings):
        settings.sample_rate = 48000
        settings.channels = 2
        settings.chunk_size = 480
        instance = AudioHandler()
        assert (instance.sample_rate, instance.channels, instance.chunk_size) == (
            48000, 2, 480,
        )

    def test_vad_aggressiveness_is_moderate(self, handler):
        assert handler.vad.aggressiveness == 2


class TestRecording:
    def test_opens_the_stream_with_the_configured_format(self, handler, fake_sd):
        handler.record_until_silence(max_duration=0.2)
        kwargs = FakeInputStream.instances[0].kwargs
        assert kwargs["samplerate"] == 16000
        assert kwargs["channels"] == 1
        assert kwargs["dtype"] == "int16"
        assert kwargs["blocksize"] == 1024

    def test_stops_at_max_duration(self, handler, fake_sd):
        audio = handler.record_until_silence(max_duration=0.5)
        expected_frames = int(0.5 * 16000 / 1024)
        assert FakeInputStream.instances[0].read_calls == expected_frames
        assert len(audio) == expected_frames * 1024

    def test_returns_int16_samples(self, handler, fake_sd):
        assert handler.record_until_silence(max_duration=0.2).dtype == np.int16

    def test_zero_duration_returns_empty_audio(self, handler, fake_sd):
        audio = handler.record_until_silence(max_duration=0.0)
        assert audio.size == 0
        assert audio.dtype == np.int16

    def test_recorded_samples_are_preserved(self, handler, fake_sd):
        FakeInputStream.scripted_blocks = [block(1024, value=1234) for _ in range(3)]
        audio = handler.record_until_silence(max_duration=0.2)
        assert int(audio.flat[0]) == 1234

    def test_recording_returns_a_flat_mono_waveform(self, handler, fake_sd):
        """faster-whisper's feature extractor wants a 1-D mono waveform."""
        audio = handler.record_until_silence(max_duration=0.2)
        assert audio.ndim == 1

    def test_multichannel_capture_is_downmixed(self, settings, fake_sd):
        settings.channels = 2
        audio = AudioHandler().record_until_silence(max_duration=0.2)
        assert audio.ndim == 1

    def test_downmix_averages_channels_without_overflowing(self):
        loud = np.full((4, 2), 32767, dtype=np.int16)
        assert AudioHandler._to_mono(loud).tolist() == [32767] * 4

    def test_stream_errors_propagate(self, handler, fake_sd, monkeypatch):
        class ExplodingStream:
            def __init__(self, **kwargs):
                raise OSError("no such device")

        monkeypatch.setattr(fake_sd, "InputStream", ExplodingStream)
        with pytest.raises(OSError, match="no such device"):
            handler.record_until_silence(max_duration=0.2)

    def test_stream_is_closed(self, handler, fake_sd):
        handler.record_until_silence(max_duration=0.2)
        assert FakeInputStream.instances[0].closed is True


class TestVoiceActivityDetection:
    """The VAD carves its own 20 ms frames, whatever the capture block size."""

    @staticmethod
    def record(handler, **kwargs):
        """Record and report what the caller can actually observe."""
        audio = handler.record_until_silence(**kwargs)
        return {
            "blocks": FakeInputStream.instances[-1].read_calls,
            "seconds": len(audio) / handler.sample_rate,
        }

    def test_frame_size_is_derived_from_the_sample_rate(self, handler):
        assert handler.vad_frame_samples == 320  # 20 ms at 16 kHz

    @pytest.mark.parametrize("chunk_size", [160, 320, 480, 1024, 2048])
    def test_silence_stops_the_recording_at_any_block_size(
        self, settings, fake_sd, fake_vad, chunk_size
    ):
        """The capture block size is independent of the VAD frame size.

        1024 samples at 16 kHz is 64 ms - a frame webrtcvad rejects outright -
        so this is the case that used to record for the full 30 s cap.
        """
        settings.chunk_size = chunk_size
        FakeVad.speech_decider = staticmethod(lambda buf: False)
        result = self.record(
            AudioHandler(), silence_threshold=0.2, max_duration=30.0
        )
        assert result["seconds"] < 1.0

    def test_continuous_speech_records_to_the_cap(self, handler, fake_sd, fake_vad):
        FakeVad.speech_decider = staticmethod(lambda buf: True)
        result = self.record(handler, silence_threshold=0.2, max_duration=0.5)
        # The cap is enforced in whole blocks, so the last partial block is
        # never started: expect the cap minus at most one block.
        block_seconds = handler.chunk_size / handler.sample_rate
        assert 0.5 - block_seconds < result["seconds"] <= 0.5

    def test_speech_resets_the_silence_counter(self, settings, fake_sd, fake_vad):
        """A blip of speech restarts the silence countdown from zero."""
        settings.chunk_size = 320  # one VAD frame per block, easy to count
        quiet = iter([False] * 200)
        FakeVad.speech_decider = staticmethod(lambda buf: next(quiet))
        uninterrupted = self.record(
            AudioHandler(), silence_threshold=0.6, max_duration=30.0
        )

        # Same run, but frame 9 is speech: the countdown restarts there.
        decisions = iter([False] * 8 + [True] + [False] * 200)
        FakeVad.speech_decider = staticmethod(lambda buf: next(decisions))
        interrupted = self.record(
            AudioHandler(), silence_threshold=0.6, max_duration=30.0
        )

        assert interrupted["blocks"] == uninterrupted["blocks"] + 9

    def test_silence_threshold_controls_how_long_we_wait(
        self, handler, fake_sd, fake_vad
    ):
        FakeVad.speech_decider = staticmethod(lambda buf: False)
        patient = self.record(handler, silence_threshold=1.5, max_duration=30.0)
        assert patient["seconds"] == pytest.approx(1.5, abs=0.1)

    def test_min_duration_keeps_a_leading_pause_from_ending_the_recording(
        self, handler, fake_sd, fake_vad
    ):
        """Someone who pauses before speaking must not get a truncated clip."""
        FakeVad.speech_decider = staticmethod(lambda buf: False)
        result = self.record(
            handler, silence_threshold=0.02, max_duration=30.0, min_duration=1.0
        )
        assert result["seconds"] >= 1.0

    def test_an_unsupported_sample_rate_disables_the_vad(self, settings, fake_sd):
        settings.sample_rate = 44100  # not one of 8/16/32/48 kHz
        instance = AudioHandler()
        assert instance.vad_frame_samples is None
        result = self.record(instance, silence_threshold=0.2, max_duration=0.2)
        assert result["seconds"] == pytest.approx(0.2, abs=0.05)

    def test_a_rejected_frame_disables_the_vad_instead_of_looping(
        self, handler, fake_sd, fake_vad
    ):
        """If the VAD ever rejects a frame it is switched off, loudly, once."""
        def explode(buf):
            raise ValueError("Error while processing frame")

        FakeVad.speech_decider = staticmethod(explode)
        result = self.record(handler, silence_threshold=0.2, max_duration=0.3)
        assert handler.vad_frame_samples is None
        assert result["seconds"] == pytest.approx(0.3, abs=0.05)


class TestPlayback:
    def test_plays_at_the_given_rate(self, handler, fake_sd):
        audio = np.zeros(1000, dtype=np.int16)
        handler.play_audio(audio, 22050)
        assert fake_sd.played[0][1] == 22050

    def test_defaults_to_the_configured_rate(self, handler, fake_sd):
        handler.play_audio(np.zeros(1000, dtype=np.int16))
        assert fake_sd.played[0][1] == 16000

    def test_blocks_until_playback_finishes(self, handler, fake_sd):
        handler.play_audio(np.zeros(1000, dtype=np.int16))
        assert fake_sd.wait_calls == 1

    def test_playback_errors_propagate(self, handler, fake_sd):
        fake_sd.play_error = OSError("device busy")
        with pytest.raises(OSError, match="device busy"):
            handler.play_audio(np.zeros(1000, dtype=np.int16))

    def test_empty_audio_is_accepted(self, handler, fake_sd):
        handler.play_audio(np.array([], dtype=np.int16), 22050)
        assert fake_sd.played[0][0].size == 0


class TestFileIO:
    def test_save_then_load_round_trip(self, handler, tmp_path):
        original = np.arange(-1000, 1000, dtype=np.int16)
        path = str(tmp_path / "clip.wav")
        handler.save_audio(original, path)
        loaded, sample_rate = handler.load_audio(path)
        assert sample_rate == 16000
        np.testing.assert_array_equal(loaded, original)

    def test_save_uses_the_configured_sample_rate(self, handler, tmp_path):
        import soundfile as sf

        path = str(tmp_path / "clip.wav")
        handler.save_audio(np.zeros(1600, dtype=np.int16), path)
        assert sf.info(path).samplerate == 16000

    def test_loading_a_missing_file_raises(self, handler, tmp_path):
        with pytest.raises((OSError, RuntimeError)):
            handler.load_audio(str(tmp_path / "nope.wav"))

    def test_saving_to_an_unwritable_path_raises(self, handler, tmp_path):
        with pytest.raises((OSError, RuntimeError)):
            handler.save_audio(
                np.zeros(10, dtype=np.int16), str(tmp_path / "no" / "dir" / "a.wav")
            )


class TestInputDevice:
    """A headless or multi-soundcard box needs to pin the capture device."""

    def test_a_device_name_is_used(self, settings, fake_sd):
        settings.audio_input_device = "USB Microphone"
        AudioHandler().record_until_silence(max_duration=0.2)
        assert FakeInputStream.instances[0].kwargs["device"] == "USB Microphone"

    def test_a_device_index_is_used(self, settings, fake_sd):
        settings.audio_input_device = "3"
        AudioHandler().record_until_silence(max_duration=0.2)
        assert FakeInputStream.instances[0].kwargs["device"] == 3

    def test_no_setting_means_the_system_default(self, settings, fake_sd):
        settings.audio_input_device = ""
        AudioHandler().record_until_silence(max_duration=0.2)
        assert FakeInputStream.instances[0].kwargs["device"] is None

    def test_whitespace_is_treated_as_unset(self, settings, fake_sd):
        settings.audio_input_device = "   "
        assert AudioHandler().input_device is None
