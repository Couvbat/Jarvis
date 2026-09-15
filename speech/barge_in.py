"""Noticing that the user has started talking while Jarvis is still answering.

The honest limitation first: a microphone in the same room as a speaker hears
the speaker. Without acoustic echo cancellation, a listener watching for
speech during playback hears Jarvis and cuts Jarvis off. So this is off by
default, and what it does when enabled is designed around the problem:

- It calibrates its noise floor on the first moments of listening, which are
  Jarvis's own voice coming back. Real speech then has to be louder than that.
- It wants speech sustained over several frames, so a door closing or a key
  press is not an interruption.

With headphones, or a microphone with hardware echo cancellation, this works
straightforwardly. On open speakers, expect to leave it off.
"""

from __future__ import annotations

import asyncio
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from audio_handler import VAD_FRAME_MS, AudioHandler

#: Consecutive speech frames before this counts as the user talking.
DEFAULT_MIN_SPEECH_FRAMES = 10  # 200 ms at 20 ms frames
#: Frames used to measure what the room sounds like, echo included.
CALIBRATION_FRAMES = 15
#: How far above the measured floor a frame must be to count as speech.
ENERGY_MARGIN = 2.5
#: Floor below which nothing counts, so silence never calibrates to zero.
MIN_ENERGY = 150.0


def frame_energy(frame: np.ndarray) -> float:
    """Root mean square of one frame."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))


class BargeInListener:
    """Watches the microphone during playback and reports sustained speech."""

    def __init__(
        self,
        audio: AudioHandler,
        min_speech_frames: int = DEFAULT_MIN_SPEECH_FRAMES,
        energy_margin: float = ENERGY_MARGIN,
    ):
        self.audio = audio
        self.min_speech_frames = min_speech_frames
        self.energy_margin = energy_margin

        self.detected = asyncio.Event()
        #: Audio captured from the moment speech started, so the words that
        #: interrupted are not lost to the turn they start.
        self.captured: Optional[np.ndarray] = None

        self._task: Optional[asyncio.Task] = None
        self._stop = False
        self._floor = MIN_ENERGY

    async def start(self) -> None:
        if self._task is not None:
            return
        self.detected.clear()
        self.captured = None
        self._stop = False
        self._task = asyncio.create_task(
            asyncio.to_thread(self._listen), name="barge-in"
        )

    async def stop(self) -> Optional[np.ndarray]:
        """Stop listening and return whatever speech was captured."""
        self._stop = True
        task, self._task = self._task, None
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:  # pragma: no cover - device-specific
                task.cancel()
            except Exception as e:  # pragma: no cover - device-specific
                logger.debug(f"Barge-in listener stopped with {e}")
        return self.captured

    def _listen(self) -> None:
        """Runs in a worker thread: the audio API is blocking."""
        frame_samples = self.audio.vad_frame_samples
        if frame_samples is None:
            logger.warning("Barge-in needs the VAD; it is disabled for this rate")
            return

        speech_run = 0
        calibration: List[float] = []
        kept: List[np.ndarray] = []

        try:
            with sd.InputStream(
                samplerate=self.audio.sample_rate,
                channels=self.audio.channels,
                dtype="int16",
                blocksize=frame_samples,
            ) as stream:
                while not self._stop:
                    block, _ = stream.read(frame_samples)
                    frame = self.audio._to_mono(block)

                    energy = frame_energy(frame)
                    if len(calibration) < CALIBRATION_FRAMES:
                        # These first frames are mostly Jarvis coming back
                        # through the microphone; that is the bar to clear.
                        calibration.append(energy)
                        if len(calibration) == CALIBRATION_FRAMES:
                            self._floor = max(MIN_ENERGY, float(np.median(calibration)))
                            logger.debug(f"Barge-in noise floor: {self._floor:.0f}")
                        continue

                    loud = energy > self._floor * self.energy_margin
                    if loud and self.audio._is_speech(frame):
                        speech_run += 1
                        kept.append(frame)
                    else:
                        speech_run = 0
                        kept.clear()

                    if speech_run >= self.min_speech_frames:
                        self.captured = np.concatenate(kept)
                        logger.info(
                            f"Barge-in: {speech_run * VAD_FRAME_MS} ms of speech "
                            f"at {energy:.0f} over a floor of {self._floor:.0f}"
                        )
                        self.detected.set()
                        return
        except Exception as e:
            # A second stream on the same device can simply be refused; that
            # costs barge-in, not the conversation.
            logger.warning(f"Barge-in listener stopped: {e}")
