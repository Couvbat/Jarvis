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

    def test_recording_keeps_the_channel_axis(self, handler, fake_sd):
        """Blocks come back as ``(frames, channels)`` and are concatenated as-is.

        The result is therefore 2-D even for mono capture - see the xfail on
        mono flattening at the bottom of this file.
        """
        audio = handler.record_until_silence(max_duration=0.2)
        assert audio.ndim == 2
        assert audio.shape[1] == 1

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
    """With a VAD-legal chunk size the silence logic behaves as designed."""

    @pytest.fixture
    def vad_sized_handler(self, settings):
        settings.chunk_size = 320  # 20 ms at 16 kHz - accepted by webrtcvad
        return AudioHandler()

    def test_silence_stops_the_recording_early(self, vad_sized_handler, fake_sd, fake_vad):
        FakeVad.speech_decider = staticmethod(lambda buf: False)
        audio = vad_sized_handler.record_until_silence(
            silence_threshold=0.2, max_duration=30.0
        )
        reads = FakeInputStream.instances[0].read_calls
        assert reads < int(30.0 * 16000 / 320)
        assert reads == 11  # 10 silent frames + the >10 frame guard
        assert len(audio) == 11 * 320

    def test_continuous_speech_records_to_the_cap(
        self, vad_sized_handler, fake_sd, fake_vad
    ):
        FakeVad.speech_decider = staticmethod(lambda buf: True)
        vad_sized_handler.record_until_silence(silence_threshold=0.2, max_duration=0.5)
        assert FakeInputStream.instances[0].read_calls == int(0.5 * 16000 / 320)

    def test_speech_resets_the_silence_counter(
        self, vad_sized_handler, fake_sd, fake_vad
    ):
        decisions = iter([False] * 8 + [True] + [False] * 100)
        FakeVad.speech_decider = staticmethod(lambda buf: next(decisions))
        vad_sized_handler.record_until_silence(
            silence_threshold=0.2, max_duration=30.0
        )
        # 8 silent + 1 speech + 10 silent = 19 reads, not 11.
        assert FakeInputStream.instances[0].read_calls == 19

    def test_short_recordings_are_not_cut_by_the_frame_guard(
        self, vad_sized_handler, fake_sd, fake_vad
    ):
        FakeVad.speech_decider = staticmethod(lambda buf: False)
        vad_sized_handler.record_until_silence(
            silence_threshold=0.01, max_duration=30.0
        )
        assert FakeInputStream.instances[0].read_calls == 11


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


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True, reason="BUG-18: the default chunk size is not a legal VAD frame"
)
def test_silence_detection_works_with_the_default_chunk_size(handler, fake_sd, fake_vad):
    """webrtcvad only accepts 10, 20 or 30 ms frames.

    CHUNK_SIZE=1024 at 16 kHz is 64 ms, so every ``is_speech`` call raises,
    the exception is swallowed as a debug log, the silence counter never
    advances and every utterance records for the full 30 s cap before Whisper
    even starts.
    """
    FakeVad.speech_decider = staticmethod(lambda buf: False)
    handler.record_until_silence(silence_threshold=0.5, max_duration=30.0)
    assert FakeInputStream.instances[0].read_calls < int(30.0 * 16000 / 1024)


@pytest.mark.xfail(strict=True, reason="BUG-19: VAD is fed multi-channel audio as-is")
def test_stereo_capture_still_feeds_mono_frames_to_the_vad(settings, fake_sd, fake_vad):
    """webrtcvad requires 16-bit *mono*.  With CHANNELS=2 the interleaved
    buffer is handed over unchanged, doubling the apparent frame length."""
    settings.channels = 2
    settings.chunk_size = 320
    instance = AudioHandler()
    FakeVad.speech_decider = staticmethod(lambda buf: False)
    instance.record_until_silence(silence_threshold=0.2, max_duration=30.0)
    assert FakeInputStream.instances[0].read_calls == 11


@pytest.mark.xfail(
    strict=True, reason="BUG-21: recorded audio is 2-D when Whisper wants 1-D"
)
def test_recording_returns_a_mono_waveform(handler, fake_sd):
    """``record_until_silence`` hands a ``(frames, 1)`` array straight to
    ``STTModule.transcribe``, which forwards it to faster-whisper.  The feature
    extractor expects a flat mono waveform."""
    audio = handler.record_until_silence(max_duration=0.2)
    assert audio.ndim == 1


@pytest.mark.xfail(strict=True, reason="BUG-20: no input device selection")
def test_input_device_can_be_configured(settings, fake_sd):
    """A headless or multi-soundcard box needs to pin the capture device;
    ``sd.InputStream`` is always opened on the system default."""
    settings.audio_input_device = "USB Microphone"
    AudioHandler().record_until_silence(max_duration=0.2)
    assert FakeInputStream.instances[0].kwargs["device"] == "USB Microphone"
