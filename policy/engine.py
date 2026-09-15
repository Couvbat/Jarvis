"""Deciding whether a tool call may happen, and who has to say so.

Three outcomes: run it, ask first, or refuse. The interesting part is *how*
the user is asked. A voice assistant cannot take "yes" for a destructive
action over the microphone: speech recognition mishears, an ambient
conversation can supply the word, and the user cannot see what they are
agreeing to. So the surface is part of the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from loguru import logger

from policy.store import ApprovalStore
from policy.taint import TaintState
from tools.schema import Risk, ToolSpec

#: Arguments longer than this are trimmed in the confirmation summary.
MAX_SUMMARY_VALUE = 80


class Surface(Enum):
    """Where a confirmation has to come from."""

    #: No confirmation needed.
    NONE = "none"
    #: Speaking is enough.
    VOICE = "voice"
    #: The user has to see it and answer at the keyboard.
    TERMINAL = "terminal"


@dataclass(frozen=True)
class Decision:
    """What the policy engine concluded about one call."""

    tool: str
    risk: Risk
    scope: str
    surface: Surface
    summary: str
    allowed: bool = True
    reason: str = ""

    @property
    def needs_confirmation(self) -> bool:
        return self.allowed and self.surface is not Surface.NONE

    @property
    def is_automatic(self) -> bool:
        return self.allowed and self.surface is Surface.NONE


def summarise(tool: str, arguments: Dict[str, Any]) -> str:
    """Render a call the way a person needs to read it before agreeing.

    The arguments are the point: "delete /home/me/Documents, recursive=true"
    is a decision someone can make, "call fs__delete" is not.
    """
    if not arguments:
        return tool
    parts = []
    for key, value in arguments.items():
        rendered = str(value)
        if len(rendered) > MAX_SUMMARY_VALUE:
            rendered = rendered[:MAX_SUMMARY_VALUE] + "…"
        parts.append(f"{key}={rendered}")
    return f"{tool} ({', '.join(parts)})"


class PolicyEngine:
    """Applies the rules in ARCHITECTURE.md section 5 to one call at a time."""

    def __init__(self, store: Optional[ApprovalStore] = None):
        self.store = store if store is not None else ApprovalStore()

    def scope_for(self, spec: ToolSpec, arguments: Dict[str, Any]) -> str:
        """What an approval for this call would cover.

        Falls back to the exact arguments, so an unfamiliar tool - an MCP one,
        say - grants the narrowest thing possible rather than a blanket yes.
        """
        if spec.scope_for is not None:
            try:
                return spec.scope_for(arguments)
            except Exception as e:
                logger.warning(f"{spec.name} could not describe its scope: {e}")
        return repr(sorted((str(k), str(v)) for k, v in arguments.items()))

    def origin_for(
        self, spec: ToolSpec, arguments: Dict[str, Any]
    ) -> Optional[str]:
        """Where a call would send data, if the tool can say."""
        if spec.origin_for is None:
            return None
        try:
            return spec.origin_for(arguments)
        except Exception as e:
            logger.warning(f"{spec.name} could not describe its origin: {e}")
            return None

    def evaluate(
        self,
        spec: ToolSpec,
        arguments: Optional[Dict[str, Any]] = None,
        taint: Optional[TaintState] = None,
    ) -> Decision:
        """Decide what has to happen before this call runs."""
        arguments = dict(arguments or {})
        taint = taint if taint is not None else TaintState()

        risk = spec.assess(arguments)
        scope = self.scope_for(spec, arguments)
        summary = summarise(spec.name, arguments)

        def decide(surface: Surface, reason: str = "", allowed: bool = True) -> Decision:
            return Decision(
                tool=spec.name, risk=risk, scope=scope,
                surface=surface, summary=summary, reason=reason, allowed=allowed,
            )

        # Refuse before asking. A prompt spends the user's attention, and
        # spending it on a call that will be refused anyway teaches them to
        # answer without reading.
        if spec.precheck is not None:
            try:
                refusal = spec.precheck(arguments)
            except Exception as e:
                logger.warning(f"{spec.name} precheck failed: {e}")
                refusal = f"could not validate the call: {e}"
            if refusal:
                return decide(Surface.NONE, refusal, allowed=False)

        # Destroying something is always shown to the user, and always at the
        # keyboard. A standing approval never covers it: deleting is exactly
        # the action you want to see every time.
        if risk >= Risk.DESTRUCTIVE:
            return decide(Surface.TERMINAL, "this destroys or overwrites data")

        # Once untrusted content is in the turn, writing is escalated
        # whatever was approved before.
        if taint.tainted and risk >= Risk.WRITE:
            return decide(Surface.TERMINAL, taint.describe())

        # Egress is escalated when it would carry that content somewhere new.
        # Reading a second note from the server that just answered is not the
        # exfiltration shape, and escalating it would make every multi-step
        # workflow a wall of prompts - which is how people learn to stop
        # reading them.
        if taint.tainted and spec.egress:
            origin = self.origin_for(spec, arguments)
            if not taint.knows_origin(origin):
                return decide(
                    Surface.TERMINAL,
                    f"{taint.describe()}, and this would reach "
                    f"{origin or 'somewhere else'}",
                )

        if spec.preapproved:
            # Authorised in configuration rather than by a prompt. The two
            # rules above still hold: this cannot wave through a destructive
            # call or one made after untrusted content entered the turn.
            return decide(Surface.NONE, "authorised in configuration")

        if self.store.is_approved(spec.name, scope, risk):
            return decide(Surface.NONE, "previously approved")

        if spec.egress:
            return decide(Surface.VOICE, "this sends a request off the machine")

        return decide(Surface.VOICE, f"{risk.name.lower().replace('_', ' ')} access")

    def remember(self, decision: Decision) -> None:
        """Record that the user approved this call for future ones like it."""
        if decision.risk >= Risk.DESTRUCTIVE:
            # Deliberately not stored: see evaluate().
            logger.info(f"Not storing an approval for destructive {decision.tool}")
            return
        self.store.approve(decision.tool, decision.scope, decision.risk)
