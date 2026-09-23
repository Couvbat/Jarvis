"""Splitting streamed text into pieces worth speaking.

The point is latency. Waiting for a whole answer before synthesising anything
puts three to eight seconds between the end of a question and the first sound;
sending each finished sentence to the synthesiser as it arrives cuts that to
roughly the time it takes to generate one sentence.
"""

from __future__ import annotations

import re
from typing import List, Optional

#: Characters that can end a spoken sentence.
TERMINATORS = ".!?…"

#: Words that end in a period without ending a sentence. A split here would
#: put a pause in the middle of "M. Dupont" and hand the synthesiser a
#: fragment starting mid-name.
ABBREVIATIONS = frozenset({
    "m", "mm", "mme", "mlle", "dr", "pr", "st", "ste", "av", "bd",
    "etc", "cf", "ex", "p", "pp", "vol", "no", "art", "fig",
    "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "approx", "dept", "est",
    "i.e", "e.g", "a.m", "p.m",
})

#: A sentence shorter than this is held back and joined to the next one, so a
#: stray "Oui." does not become its own audio clip. The first piece of an
#: answer is exempt: that one is the whole point of the exercise.
MIN_CHARS = 20


def _is_abbreviation(text: str) -> bool:
    """True when the period at the end of ``text`` is part of a word."""
    match = re.search(r"([\w.]+)\.$", text, re.UNICODE)
    if not match:
        return False
    return match.group(1).lower().rstrip(".") in ABBREVIATIONS


def _splits_here(buffer: str, index: int) -> bool:
    """Whether the terminator at ``index`` really ends a sentence."""
    char = buffer[index]
    if char not in TERMINATORS:
        return False

    following = buffer[index + 1:index + 2]
    # Only a terminator followed by whitespace is a boundary we can trust
    # mid-stream: "example.com" and "3.14" have no space after the dot.
    if following and not following.isspace():
        return False
    if not following:
        return False  # more may still arrive; wait for it

    if char == "." and _is_abbreviation(buffer[: index + 1]):
        return False
    if char == "." and re.search(r"\d\.$", buffer[: index + 1]):
        # "1." mid-number, or a numbered list item: not a pause.
        return False
    return True


class SentenceChunker:
    """Accumulates streamed text and hands back pieces ready to speak."""

    def __init__(self, min_chars: int = MIN_CHARS):
        self.min_chars = min_chars
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> List[str]:
        """Add newly generated text; return whatever is ready to speak."""
        self._buffer += text
        ready: List[str] = []

        while True:
            piece = self._take_next()
            if piece is None:
                break
            ready.append(piece)

        return ready

    def _take_next(self) -> Optional[str]:
        # A newline is a hard break: a list item or paragraph is its own
        # utterance whatever its length.
        newline = self._buffer.find("\n")

        boundary = -1
        for index in range(len(self._buffer)):
            if _splits_here(self._buffer, index):
                boundary = index
                break

        if newline != -1 and (boundary == -1 or newline < boundary):
            piece = self._buffer[:newline].strip()
            self._buffer = self._buffer[newline + 1:]
            if not piece:
                return self._take_next()
            self._emitted += 1
            return piece

        if boundary == -1:
            return None

        piece = self._buffer[: boundary + 1].strip()
        remainder = self._buffer[boundary + 1:]

        # Too short to be worth its own clip - unless it is the first thing
        # said, where getting any sound out quickly is the whole point.
        if self._emitted > 0 and len(piece) < self.min_chars:
            if not remainder.strip():
                return None  # nothing to join it to yet
            joined = self._take_more(piece, remainder)
            if joined is None:
                return None
            return joined

        self._buffer = remainder
        self._emitted += 1
        return piece

    def _take_more(self, piece: str, remainder: str) -> Optional[str]:
        """Extend a too-short piece with the sentence that follows it."""
        self._buffer = remainder
        following = self._take_next()
        if following is None:
            self._buffer = piece + remainder
            return None
        return f"{piece} {following}"

    def flush(self) -> Optional[str]:
        """Whatever is left once generation has finished."""
        piece = self._buffer.strip()
        self._buffer = ""
        if piece:
            self._emitted += 1
        return piece or None

    def reset(self) -> None:
        self._buffer = ""
        self._emitted = 0
