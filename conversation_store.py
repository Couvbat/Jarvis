"""Conversations kept across restarts.

The turns are already structured - that was the work in Phase 0 - so storing
them is mostly a matter of writing them down. What it buys is the ability to
pick up where a session left off, and it is the prerequisite for anything that
wants to look back: recalling an earlier answer, usage analytics, per-user
memory.

Conversations are written to disk in the clear, on the user's own machine.
Set CONVERSATION_HISTORY=false to keep nothing.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at   TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    position        INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    name            TEXT,
    tool_calls      TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_by_conversation
    ON messages (conversation_id, position);
"""


@dataclass(frozen=True)
class Conversation:
    """One session's worth of turns."""

    id: int
    started_at: str
    ended_at: str | None
    message_count: int = 0
    preview: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    """Writes turns down and reads them back."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            resolved = Path(self.path).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(resolved)

        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(SCHEMA)

    def close(self) -> None:
        self._connection.close()

    # -- writing ---------------------------------------------------------- #

    def start(self) -> int:
        """Open a new conversation and return its id."""
        with self._connection:
            cursor = self._connection.execute(
                "INSERT INTO conversations (started_at) VALUES (?)", (_now(),)
            )
        return int(cursor.lastrowid)

    def append(self, conversation_id: int, message: dict[str, Any]) -> None:
        """Record one turn. Never raises: losing the log is not worth a crash."""
        try:
            with self._connection:
                position = self._connection.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM messages "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]

                tool_calls = message.get("tool_calls")
                self._connection.execute(
                    "INSERT INTO messages "
                    "(conversation_id, position, role, content, name, tool_calls, "
                    " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        conversation_id,
                        position,
                        str(message.get("role", "")),
                        str(message.get("content") or ""),
                        message.get("name"),
                        json.dumps(tool_calls) if tool_calls else None,
                        _now(),
                    ),
                )
        except sqlite3.Error as e:
            logger.error(f"Could not record a turn: {e}")

    def end(self, conversation_id: int) -> None:
        """Mark a conversation finished."""
        try:
            with self._connection:
                self._connection.execute(
                    "UPDATE conversations SET ended_at = ? WHERE id = ?",
                    (_now(), conversation_id),
                )
        except sqlite3.Error as e:  # pragma: no cover - defensive
            logger.error(f"Could not close conversation {conversation_id}: {e}")

    # -- reading ---------------------------------------------------------- #

    def messages(self, conversation_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        """The turns of one conversation, oldest first.

        ``limit`` keeps the *most recent* turns, which is what resuming wants:
        the end of a conversation is the part still worth carrying.
        """
        columns = "position, role, content, name, tool_calls"
        if limit is None:
            query = (
                f"SELECT {columns} FROM messages "
                "WHERE conversation_id = ? ORDER BY position"
            )
            parameters: tuple = (conversation_id,)
        else:
            # Take the newest rows, then put them back in order. The subquery
            # has to carry `position` for the outer sort to have anything to
            # work with.
            query = (
                f"SELECT {columns} FROM ("
                f"  SELECT {columns} FROM messages WHERE conversation_id = ? "
                "   ORDER BY position DESC LIMIT ?"
                ") ORDER BY position"
            )
            parameters = (conversation_id, limit)

        with closing(self._connection.execute(query, parameters)) as cursor:
            rows = cursor.fetchall()

        messages = []
        for row in rows:
            message: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["name"]:
                message["name"] = row["name"]
            if row["tool_calls"]:
                try:
                    message["tool_calls"] = json.loads(row["tool_calls"])
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    pass
            messages.append(message)
        return messages

    def recent(self, limit: int = 10) -> list[Conversation]:
        """The most recent conversations, newest first."""
        with closing(self._connection.execute(
            "SELECT c.id, c.started_at, c.ended_at, "
            "       COUNT(m.id) AS message_count, "
            "       COALESCE(MIN(CASE WHEN m.role = 'user' THEN m.content END), '') "
            "         AS preview "
            "FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id "
            "GROUP BY c.id ORDER BY c.started_at DESC, c.id DESC LIMIT ?",
            (limit,),
        )) as cursor:
            return [
                Conversation(
                    id=int(row["id"]),
                    started_at=row["started_at"],
                    ended_at=row["ended_at"],
                    message_count=int(row["message_count"]),
                    preview=row["preview"] or "",
                )
                for row in cursor.fetchall()
            ]

    def last_id(self) -> int | None:
        """The most recent conversation that actually has turns in it."""
        with closing(self._connection.execute(
            "SELECT c.id FROM conversations c "
            "JOIN messages m ON m.conversation_id = c.id "
            "GROUP BY c.id ORDER BY c.started_at DESC, c.id DESC LIMIT 1"
        )) as cursor:
            row = cursor.fetchone()
        return int(row["id"]) if row else None

    def search(self, text: str, limit: int = 20) -> list[dict[str, Any]]:
        """Turns containing some text, newest first."""
        needle = f"%{text}%"
        with closing(self._connection.execute(
            "SELECT conversation_id, role, content, created_at FROM messages "
            "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
            (needle, limit),
        )) as cursor:
            return [dict(row) for row in cursor.fetchall()]

    def purge(self) -> int:
        """Forget everything."""
        with self._connection:
            self._connection.execute("DELETE FROM messages")
            cursor = self._connection.execute("DELETE FROM conversations")
        return cursor.rowcount
