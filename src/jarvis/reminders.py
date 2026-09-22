"""Reminders that survive a restart.

"Remind me to call the plumber tomorrow at nine" is the thing people expect
of an assistant and notice the absence of. Two decisions shape this:

**No date-parsing library.** `dateparser` and friends exist to turn "demain à
9h" into a timestamp, but there is already a language model in the loop whose
whole job is understanding what was said. So the tool takes an ISO timestamp
or a number of minutes, the current local time goes in the system prompt, and
the model does the conversion. What the tool does instead is *check* the
result - a reminder in the past, or in the year 3000, is a model that got the
arithmetic wrong, and saying so lets it try again.

**Reminders fire between turns, not during one.** Speaking over a recording
would put Jarvis's own voice into the microphone, which is the problem
barge-in exists for and is off by default because of. So a due reminder
interrupts the wait for the wake word, or is delivered when the current turn
ends. The cost is that a reminder can be a few seconds late, which nobody
notices, and the alternative is an assistant that talks over you.

A reminder that came due while Jarvis was not running is delivered at the
next startup, late and labelled as such, rather than silently dropped.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    due_at     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    fired_at   TEXT,
    cancelled  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS reminders_by_due ON reminders (due_at);
"""

#: Nothing may be scheduled further out than this. A model that computed
#: "in 5 minutes" as the year 3000 should be told, not obeyed.
MAX_HORIZON = timedelta(days=366)
#: How long after its time a reminder is still delivered rather than dropped.
#: Longer than a working day, so one set on Friday evening still arrives on
#: Monday morning.
MAX_LATENESS = timedelta(days=7)


@dataclass(frozen=True)
class Reminder:
    """One scheduled reminder."""

    id: int
    text: str
    due_at: datetime
    created_at: datetime
    fired_at: datetime | None = None

    def describe(self, now: datetime | None = None) -> str:
        """How this reads when spoken back."""
        moment = self.due_at.astimezone()
        reference = (now or datetime.now(timezone.utc)).astimezone()

        if moment.date() == reference.date():
            when = f"today at {moment:%H:%M}"
        elif moment.date() == (reference + timedelta(days=1)).date():
            when = f"tomorrow at {moment:%H:%M}"
        else:
            when = f"{moment:%A %d %B at %H:%M}"
        return f"{self.text} - {when}"


class ReminderStore:
    """Reminders on disk, so a restart does not forget them."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.executescript(SCHEMA)
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def _row(row: tuple) -> Reminder:
        return Reminder(
            id=row[0],
            text=row[1],
            due_at=datetime.fromisoformat(row[2]),
            created_at=datetime.fromisoformat(row[3]),
            fired_at=datetime.fromisoformat(row[4]) if row[4] else None,
        )

    # -- writing ------------------------------------------------------------ #

    def add(self, text: str, due_at: datetime) -> Reminder:
        """Store a reminder. The caller has already validated the time."""
        now = datetime.now(timezone.utc)
        with closing(self._connect()) as db:
            cursor = db.execute(
                "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
                (text, due_at.astimezone(timezone.utc).isoformat(), now.isoformat()),
            )
            db.commit()
            reminder_id = cursor.lastrowid

        logger.info(f"Reminder {reminder_id} set for {due_at.astimezone():%c}: {text}")
        return Reminder(reminder_id, text, due_at, now)

    def cancel(self, reminder_id: int) -> bool:
        """Cancel one reminder. False when there was nothing to cancel."""
        with closing(self._connect()) as db:
            changed = db.execute(
                "UPDATE reminders SET cancelled = 1 "
                "WHERE id = ? AND cancelled = 0 AND fired_at IS NULL",
                (reminder_id,),
            ).rowcount
            db.commit()
        return bool(changed)

    def mark_fired(self, reminder_id: int) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "UPDATE reminders SET fired_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), reminder_id),
            )
            db.commit()

    # -- reading ------------------------------------------------------------ #

    def pending(self) -> list[Reminder]:
        """Everything still waiting to fire, soonest first."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT id, text, due_at, created_at, fired_at FROM reminders "
                "WHERE cancelled = 0 AND fired_at IS NULL ORDER BY due_at"
            ).fetchall()
        return [self._row(row) for row in rows]

    def due(self, now: datetime | None = None) -> list[Reminder]:
        """Reminders whose time has come and which have not fired.

        A reminder from a month ago is not delivered: waking up to a week of
        backlog would be worse than losing it, and it is still visible in the
        list until it is cancelled.
        """
        moment = now or datetime.now(timezone.utc)
        floor = moment - MAX_LATENESS
        return [
            reminder for reminder in self.pending()
            if floor <= reminder.due_at <= moment
        ]

    def next_due(self) -> datetime | None:
        """When the soonest pending reminder is due."""
        pending = self.pending()
        return pending[0].due_at if pending else None


def resolve_due_time(
    when: str | None = None,
    in_minutes: float | int | None = None,
    now: datetime | None = None,
) -> datetime:
    """Work out when a reminder should fire, or say why it cannot.

    Two ways in, because a model is reliable at "in 30 minutes" and less so at
    turning "tomorrow at nine" into an ISO timestamp. Raising ValueError with
    the reason lets it correct the call rather than setting a wrong reminder.
    """
    moment = (now or datetime.now(timezone.utc)).astimezone()

    if in_minutes is not None:
        try:
            minutes = float(in_minutes)
        except (TypeError, ValueError) as e:
            raise ValueError(f"'{in_minutes}' is not a number of minutes") from e
        if minutes <= 0:
            raise ValueError("a reminder has to be in the future")
        due = moment + timedelta(minutes=minutes)

    elif when:
        text = str(when).strip().replace("Z", "+00:00")
        try:
            due = datetime.fromisoformat(text)
        except ValueError as e:
            raise ValueError(
                f"'{when}' is not a date and time I can read; use "
                f"YYYY-MM-DDTHH:MM, or give a number of minutes instead"
            ) from e
        if due.tzinfo is None:
            # A model that writes "2026-09-23T09:00" means nine o'clock here,
            # not nine o'clock UTC.
            due = due.astimezone()

    else:
        raise ValueError("say when: either a date and time, or a number of minutes")

    if due <= moment:
        raise ValueError(
            f"{due.astimezone():%Y-%m-%d %H:%M} has already passed - "
            f"it is {moment:%Y-%m-%d %H:%M} now"
        )
    if due - moment > MAX_HORIZON:
        raise ValueError(
            f"{due.astimezone():%Y-%m-%d %H:%M} is more than a year away; "
            f"that is probably not what was meant"
        )
    return due
