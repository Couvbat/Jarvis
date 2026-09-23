"""Tracking untrusted content within a turn.

Jarvis combines the three things that make prompt injection worth attempting:
access to private files, ingestion of content nobody vetted, and a way to send
data back out. A local model resists injection *less* well than a frontier
one, so the defence cannot be the model.

This does not try to detect injection. It cuts the chain between reading
hostile content and acting on it unseen: once a turn has taken in untrusted
content, anything that writes or leaves the machine needs the user to look at
it, whatever they approved before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from jarvis.tools.schema import ToolResult


@dataclass
class TaintState:
    """Whether the current turn has ingested content the user did not write."""

    tainted: bool = False
    sources: list[str] = field(default_factory=list)
    #: The places the untrusted content came from: a domain, a server name.
    origins: list[str] = field(default_factory=list)

    def observe(self, tool_name: str, result: ToolResult) -> None:
        """Record a tool result, marking the turn if it came from outside."""
        if result.untrusted:
            if not self.tainted:
                logger.info(f"Turn tainted by untrusted content from {tool_name}")
            self.tainted = True
            if tool_name not in self.sources:
                self.sources.append(tool_name)
            origin = result.origin or tool_name
            if origin not in self.origins:
                self.origins.append(origin)

    def knows_origin(self, origin: str | None) -> bool:
        """True when this turn's untrusted content already came from there.

        Reading more from a source that already tainted the turn is not the
        shape worth escalating; handing its content somewhere *new* is.
        """
        return origin is not None and origin in self.origins

    def describe(self) -> str:
        """Why the turn is tainted, for the confirmation prompt."""
        if not self.tainted:
            return ""
        return f"this turn read untrusted content from {', '.join(self.sources)}"

    def reset(self) -> None:
        """Start a fresh turn. Taint does not carry over."""
        self.tainted = False
        self.sources = []
        self.origins = []
