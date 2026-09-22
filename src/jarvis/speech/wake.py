"""Waiting to be spoken to, rather than transcribing the whole room.

Without this, every noise in the room is recorded, transcribed and sent to
the model. That is not only wasteful, it is the wrong behaviour for something
that lives on a desk: an assistant that answers a conversation it was not part
of is worse than one that misses a command.

The detector is openWakeWord, which runs locally and ships a pretrained
`hey_jarvis` model. Porcupine, which `FEATURES_IDEA.md` suggested, needs a
Picovoice API key - a key check against a remote service is exactly what "100%
local" rules out, however good the detector is.

Two details decide whether this is pleasant or infuriating to use:

- **The command that follows in the same breath is kept.** People say "hey
  Jarvis quelle heure est-il", not "hey Jarvis", pause, "quelle heure est-il".
  So detection returns the audio that came after the phrase, and it is
  prepended to the recording - the same mechanism barge-in uses for the words
  spoken before it noticed.
- **A follow-up does not need the phrase again.** Saying the wake word before
  every turn of a five-turn exchange is the thing people stop doing. After an
  answer there is a short window where Jarvis simply listens.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

#: openWakeWord's models consume 80 ms of 16 kHz mono audio at a time. Other
#: sizes are accepted but buffered internally, which only adds latency; the
#: frames are carved here instead so what reaches the model is predictable.
FRAME_SAMPLES = 1280
#: The rate those models were trained at. Nothing else is meaningful.
MODEL_SAMPLE_RATE = 16000

INSTALL_HINT = (
    'pip install "jarvis-assistant[wakeword]", then '
    "python -c \"import openwakeword.utils; openwakeword.utils.download_models()\""
)


class WakeWordUnavailable(RuntimeError):
    """openWakeWord is not installed, or its models are not downloaded.

    Raised at construction, never during listening: the caller can then fall
    back to the always-listening loop and say so once, instead of failing a
    turn at a time.
    """


class WakeWord:
    """Reports when the wake phrase has just been said.

    Audio is pushed in with :meth:`feed`, which answers for each frame. The
    object holds the model's streaming state, so one instance belongs to one
    listening session; :meth:`reset` clears it after a detection so the same
    words cannot fire twice.
    """

    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        sample_rate: int = MODEL_SAMPLE_RATE,
        detector=None,
    ):
        if sample_rate != MODEL_SAMPLE_RATE:
            raise WakeWordUnavailable(
                f"Wake word detection needs {MODEL_SAMPLE_RATE} Hz audio, but "
                f"SAMPLE_RATE is {sample_rate}. Lower it, or turn WAKE_WORD off."
            )

        self.model_name = model
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._pending = np.empty(0, dtype=np.int16)
        self._detector = detector if detector is not None else self._load(model)

    @staticmethod
    def _load(model: str):
        """Build the openWakeWord model, or explain how to get one.

        Imported here rather than at module scope: the package pulls in
        onnxruntime and tflite-runtime, which is a lot to ask of someone who
        types their commands.
        """
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeWordUnavailable(
                f"openWakeWord is not installed. {INSTALL_HINT}"
            ) from e

        try:
            return Model(wakeword_models=[model])
        except Exception as e:
            # Almost always the models not being downloaded yet, which is a
            # separate step from installing the package and worth saying.
            raise WakeWordUnavailable(
                f"Could not load the wake word model '{model}': {e}. {INSTALL_HINT}"
            ) from e

    # -- listening ---------------------------------------------------------- #

    def feed(self, audio: np.ndarray) -> bool:
        """Push mono audio in; True when the phrase was heard in it.

        Any block size is accepted. Whatever does not fill a frame is held
        until the next call, so a detection is never lost at a block boundary.
        """
        if audio is None or len(audio) == 0:
            return False

        self._pending = np.concatenate([self._pending, np.asarray(audio).ravel()])

        heard = False
        while len(self._pending) >= FRAME_SAMPLES:
            frame = self._pending[:FRAME_SAMPLES]
            self._pending = self._pending[FRAME_SAMPLES:]
            if self._score(frame) >= self.threshold:
                heard = True
                # Keep draining: the buffered frames after the phrase are part
                # of the command, and feeding them to a reset model next time
                # is better than scoring them against a model that has already
                # fired.
        return heard

    def _score(self, frame: np.ndarray) -> float:
        """The model's confidence for one frame, or 0 if it could not say."""
        try:
            scores = self._detector.predict(frame)
        except Exception as e:
            # A detector that fails on one frame fails on all of them, but
            # crashing the listening loop would take the assistant down with
            # it. Report and treat the frame as silence.
            logger.warning(f"Wake word detection failed: {e}")
            return 0.0

        if not scores:
            return 0.0
        # predict() answers for every loaded model; one is loaded, but reading
        # the maximum is correct either way and does not depend on the name
        # openWakeWord chose for the file.
        return max(float(value) for value in scores.values())

    def reset(self) -> None:
        """Forget what has been heard so far."""
        self._pending = np.empty(0, dtype=np.int16)
        reset = getattr(self._detector, "reset", None)
        if reset is not None:
            try:
                reset()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Could not reset the wake word detector: {e}")

    def describe(self) -> str:
        return f"wake word '{self.model_name}' (threshold {self.threshold})"
