"""Setting, listing and cancelling reminders.

These are the only tools Jarvis has that do something *later*. That makes the
confirmation question different from the rest: a file write is visible the
moment it happens, while a reminder is a promise about a time nobody is
watching. Setting one is cheap and reversible, so it is WRITE rather than
DESTRUCTIVE - but cancelling one destroys something the user asked for and
cannot be recovered, so that is.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from jarvis.reminders import ReminderStore, resolve_due_time
from jarvis.tools.schema import Risk, ToolResult, ToolSpec, namespaced

NAMESPACE = "remind"

#: Beyond this, a list is read out rather than listened to.
MAX_LISTED = 20


class ScheduleTools:
    """The reminder side of the toolbox."""

    def __init__(self, store: ReminderStore):
        self.store = store

    def create(
        self,
        text: str = "",
        when: str = "",
        in_minutes: Any = None,
        **_: Any,
    ) -> ToolResult:
        """Set a reminder."""
        what = (text or "").strip()
        if not what:
            return ToolResult.error("a reminder needs something to remind about")

        try:
            due = resolve_due_time(when=when or None, in_minutes=in_minutes)
        except ValueError as e:
            # Handed back so the model can correct the call rather than
            # setting a reminder for the wrong moment.
            return ToolResult.error(str(e))

        try:
            reminder = self.store.add(what, due)
        except Exception as e:
            logger.error(f"Could not store a reminder: {e}")
            return ToolResult.error(f"could not save the reminder: {e}")

        return ToolResult(f"Reminder set: {reminder.describe()}")

    def list(self, **_: Any) -> ToolResult:
        """What is still scheduled."""
        try:
            pending = self.store.pending()
        except Exception as e:
            logger.error(f"Could not read the reminders: {e}")
            return ToolResult.error(f"could not read the reminders: {e}")

        if not pending:
            return ToolResult("There are no reminders set.")

        now = datetime.now(timezone.utc)
        lines = [
            f"{reminder.id}. {reminder.describe(now)}"
            for reminder in pending[:MAX_LISTED]
        ]
        if len(pending) > MAX_LISTED:
            lines.append(f"...and {len(pending) - MAX_LISTED} more.")
        return ToolResult("\n".join(lines))

    def cancel(self, reminder_id: Any = None, **_: Any) -> ToolResult:
        """Cancel one reminder by its number."""
        try:
            identifier = int(reminder_id)
        except (TypeError, ValueError):
            return ToolResult.error(
                "which reminder? Use the number from the reminder list"
            )

        if self.store.cancel(identifier):
            return ToolResult(f"Reminder {identifier} cancelled.")
        return ToolResult.error(
            f"there is no reminder {identifier} still waiting to fire"
        )


def build_tools(store: ReminderStore) -> list[ToolSpec]:
    """Build the reminder tools."""
    tools = ScheduleTools(store)
    return [
        ToolSpec(
            name=namespaced(NAMESPACE, "set"),
            description=(
                "Set a reminder for later. Give either 'when' as a date and "
                "time in YYYY-MM-DDTHH:MM form, or 'in_minutes' as a number "
                "of minutes from now - whichever the user's words map to more "
                "directly. The current date and time are in your instructions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "What to remind the user about",
                    },
                    "when": {
                        "type": "string",
                        "description": (
                            "Local date and time, YYYY-MM-DDTHH:MM "
                            "(for example 2026-09-23T09:00)"
                        ),
                    },
                    "in_minutes": {
                        "type": "number",
                        "description": "Minutes from now, instead of 'when'",
                    },
                },
                "required": ["text"],
            },
            handler=tools.create,
            # Nothing on disk is at stake and it can be cancelled, but it is a
            # commitment made on the user's behalf, so they are told.
            risk=Risk.WRITE,
            scope_for=lambda arguments: "reminders",
        ),
        ToolSpec(
            name=namespaced(NAMESPACE, "list"),
            description="List the reminders that are still scheduled.",
            input_schema={"type": "object", "properties": {}},
            handler=tools.list,
            risk=Risk.READ_ONLY,
            # The user's own reminders, set through this same assistant.
            preapproved=True,
            scope_for=lambda arguments: "reminders",
        ),
        ToolSpec(
            name=namespaced(NAMESPACE, "cancel"),
            description=(
                "Cancel a scheduled reminder, by the number shown when the "
                "reminders are listed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "integer",
                        "description": "The reminder's number",
                    },
                },
                "required": ["reminder_id"],
            },
            handler=tools.cancel,
            # A cancelled reminder is a promise quietly dropped, and there is
            # no way to notice until the moment it should have fired.
            risk=Risk.DESTRUCTIVE,
            scope_for=lambda arguments: "reminders",
        ),
    ]
