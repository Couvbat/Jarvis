"""Audio input/output handler with Voice Activity Detection."""


import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from loguru import logger

from config import settings

# webrtcvad only accepts frames of exactly 10, 20 or 30 ms of 16-bit mono PCM.
# The capture block size is a separate concern (latency vs. syscall overhead),
# so frames are carved out of whatever blocks the input stream delivers.
VAD_FRAME_MS = 20
VAD_SAMPLE_RATES = (8000, 16000, 32000, 48000)


class AudioHandler:
    """Handles audio recording and playback with VAD support."""

    def __init__(self):
        self.sample_rate = settings.sample_rate
        self.channels = settings.channels
        self.chunk_size = settings.chunk_size
        self.input_device = self._resolve_device(settings.audio_input_device)
        self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3, 2 is moderate
        self.vad_frame_samples = self._vad_frame_samples()

    @staticmethod
    def _resolve_device(setting: str):
        """A device name, an index, or None for the system default."""
        value = (setting or "").strip()
        if not value:
            return None
        return int(value) if value.isdigit() else value

    def _vad_frame_samples(self) -> int | None:
        """Samples per VAD frame, or None when the rate rules the VAD out."""
        if self.sample_rate not in VAD_SAMPLE_RATES:
            logger.warning(
                f"Sample rate {self.sample_rate} Hz is not supported by webrtcvad "
                f"{VAD_SAMPLE_RATES}; silence detection is disabled and recordings "
                f"will run to max_duration."
            )
            return None
        return self.sample_rate * VAD_FRAME_MS // 1000

    @staticmethod
    def _to_mono(block: np.ndarray) -> np.ndarray:
        """Downmix a (frames, channels) block to a flat mono waveform."""
        if block.ndim == 1:
            return block
        if block.shape[1] == 1:
            return block[:, 0]
        # int32 intermediate so the average cannot overflow int16.
        return block.astype(np.int32).mean(axis=1).astype(np.int16)

    def _is_speech(self, frame: np.ndarray) -> bool:
        """Run the VAD on one mono frame, disabling it if it ever rejects one."""
        try:
            return self.vad.is_speech(frame.tobytes(), self.sample_rate)
        except Exception as e:
            # Logged once, not per frame: a VAD that rejects one frame rejects
            # them all, and silently swallowing it costs a full max_duration
            # recording on every utterance.
            logger.warning(f"VAD rejected a frame ({e}); silence detection disabled")
            self.vad_frame_samples = None
            return True

    def record_until_silence(
        self,
        silence_threshold: float = 1.0,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        prefix: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Record audio until the speaker falls silent.

        Args:
            silence_threshold: Seconds of continuous silence before stopping
            max_duration: Hard cap on the recording length in seconds
            min_duration: Never stop on silence before this many seconds, so a
                pause before the first word does not end the recording
            prefix: Audio already captured, prepended to the result. Barge-in
                hears the first words before this recording starts, and losing
                them would cost the user the beginning of their sentence.

        Returns:
            Mono audio as a flat int16 numpy array
        """
        logger.info("Starting audio recording...")

        blocks = []
        pending = np.empty(0, dtype=np.int16)  # mono samples not yet framed
        silence_run = 0
        recorded_samples = 0

        silence_frames_needed = max(1, round(silence_threshold * 1000 / VAD_FRAME_MS))
        min_samples = int(min_duration * self.sample_rate)
        max_blocks = int(max_duration * self.sample_rate / self.chunk_size)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                blocksize=self.chunk_size,
                device=self.input_device,
            ) as stream:
                logger.info("Listening... (speak now)")

                for _ in range(max_blocks):
                    audio_chunk, _ = stream.read(self.chunk_size)
                    blocks.append(audio_chunk.copy())
                    recorded_samples += len(audio_chunk)

                    frame_samples = self.vad_frame_samples
                    if frame_samples is None:
                        continue

                    # Blocks are whatever size the stream delivers; the VAD gets
                    # exact 20 ms mono frames carved out of the running buffer.
                    pending = np.concatenate([pending, self._to_mono(audio_chunk)])
                    while len(pending) >= frame_samples:
                        frame = pending[:frame_samples]
                        pending = pending[frame_samples:]

                        if self._is_speech(frame):
                            silence_run = 0
                        else:
                            silence_run += 1

                        if self.vad_frame_samples is None:
                            break  # _is_speech just switched the VAD off

                    if (
                        silence_run >= silence_frames_needed
                        and recorded_samples >= min_samples
                    ):
                        logger.info("Silence detected, stopping recording")
                        break

        except Exception as e:
            logger.error(f"Recording error: {e}")
            raise

        head = self._to_mono(prefix) if prefix is not None and len(prefix) else None

        if not blocks:
            return head if head is not None else np.array([], dtype=np.int16)

        audio_data = self._to_mono(np.concatenate(blocks, axis=0))
        if head is not None:
            audio_data = np.concatenate([head, audio_data])
        logger.info(f"Recording complete: {len(audio_data) / self.sample_rate:.2f}s")

        return audio_data

    def play_audio(self, audio_data: np.ndarray, sample_rate: int | None = None):
        """
        Play audio data.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate (uses default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        logger.info(f"Playing audio: {len(audio_data) / sample_rate:.2f}s")

        try:
            sd.play(audio_data, sample_rate)
            sd.wait()
            logger.debug("Playback complete")
        except Exception as e:
            logger.error(f"Playback error: {e}")
            raise

    def stop_playback(self):
        """Cut playback immediately, for an interruption."""
        try:
            sd.stop()
        except Exception as e:
            logger.debug(f"Could not stop playback: {e}")

    def save_audio(self, audio_data: np.ndarray, filename: str):
        """Save audio data to file."""
        try:
            sf.write(filename, audio_data, self.sample_rate)
            logger.info(f"Audio saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise

    def load_audio(self, filename: str) -> tuple[np.ndarray, int]:
        """Load audio from file."""
        try:
            audio_data, sample_rate = sf.read(filename, dtype='int16')
            logger.info(f"Audio loaded from {filename}")
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
