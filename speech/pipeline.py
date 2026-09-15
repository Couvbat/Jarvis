"""Speaking sentences in order while later ones are still being made.

Two stages rather than one loop: a synthesiser and a player, joined by a small
queue. With a single loop, nothing is being synthesised while something is
playing, so every sentence boundary costs an audible gap. Overlapping the two
means the next clip is usually ready before the current one ends.

The pipeline is also where an interruption lands. Stopping mid-answer means
dropping what has not played yet and cutting what is playing right now, which
only works if somebody owns the queue.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional, Tuple

from loguru import logger

#: Clips held ready ahead of playback. Two is enough to cover a synthesis that
#: takes longer than the clip before it, without holding much audio in memory.
AUDIO_QUEUE_SIZE = 2

_STOP = object()


class SpeechPipeline:
    """Queues text, synthesises it, and plays the results in order."""

    def __init__(
        self,
        tts: Any,
        audio: Any,
        on_fallback: Optional[Callable[[str], None]] = None,
    ):
        self.tts = tts
        self.audio = audio
        #: Called with text that could not be spoken, so it can still be shown.
        self.on_fallback = on_fallback

        self._text: asyncio.Queue = asyncio.Queue()
        self._audio: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE_SIZE)
        self._workers: List[asyncio.Task] = []
        self._interrupted = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._pending = 0
        self.spoken: List[str] = []

    # -- lifecycle -------------------------------------------------------- #

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._synthesise_loop(), name="speech:synthesise"),
            asyncio.create_task(self._play_loop(), name="speech:play"),
        ]

    async def stop(self) -> None:
        if not self._workers:
            return
        await self._text.put(_STOP)
        await self._audio.put(_STOP)
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def __aenter__(self) -> "SpeechPipeline":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    # -- queueing --------------------------------------------------------- #

    def say(self, text: str) -> None:
        """Queue one piece of text. Returns at once; speaking happens behind."""
        if not text or not text.strip():
            return
        self._pending += 1
        self._idle.clear()
        self._text.put_nowait(text)

    async def drain(self) -> None:
        """Wait until everything queued has been spoken or dropped."""
        await self._idle.wait()

    async def interrupt(self) -> None:
        """Stop now: cut what is playing and drop what has not played."""
        self._interrupted.set()
        try:
            await asyncio.to_thread(self.audio.stop_playback)
        except Exception as e:  # pragma: no cover - device-specific
            logger.debug(f"Could not stop playback: {e}")

        dropped = self._drain_queue(self._text) + self._drain_queue(self._audio)
        if dropped:
            logger.info(f"Interrupted: dropped {dropped} queued utterance(s)")
        self._settle(dropped)

    def resume(self) -> None:
        """Accept new speech again after an interruption."""
        self._interrupted.clear()

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    # -- internals -------------------------------------------------------- #

    def _drain_queue(self, queue: asyncio.Queue) -> int:
        dropped = 0
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            queue.task_done()
            if item is not _STOP:
                dropped += 1

    def _settle(self, finished: int = 1) -> None:
        """Account for utterances that have left the pipeline."""
        self._pending = max(0, self._pending - finished)
        if self._pending == 0:
            self._idle.set()

    async def _synthesise_loop(self) -> None:
        while True:
            text = await self._text.get()
            self._text.task_done()
            if text is _STOP:
                return

            if self._interrupted.is_set():
                self._settle()
                continue

            try:
                clip = await asyncio.to_thread(self.tts.synthesize, text)
            except Exception as e:
                logger.error(f"TTS error: {e}")
                if self.on_fallback:
                    self.on_fallback(text)
                self._settle()
                continue

            audio, sample_rate = clip
            if len(audio) == 0:
                if self.on_fallback:
                    self.on_fallback(text)
                self._settle()
                continue

            await self._audio.put((text, audio, sample_rate))

    async def _play_loop(self) -> None:
        while True:
            item: Tuple[str, Any, int] = await self._audio.get()
            self._audio.task_done()
            if item is _STOP:
                return

            text, audio, sample_rate = item
            if self._interrupted.is_set():
                self._settle()
                continue

            try:
                await asyncio.to_thread(self.audio.play_audio, audio, sample_rate)
                self.spoken.append(text)
            except Exception as e:
                logger.error(f"Playback error: {e}")
                if self.on_fallback:
                    self.on_fallback(text)
            finally:
                self._settle()
