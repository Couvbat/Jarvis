"""Approvals the user has granted, kept across restarts.

Replaces command_whitelist.json. SQLite rather than a JSON file because the
old store rewrote the whole file on every change and swallowed the error if
that failed, so a full disk lost approvals silently.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from tools.schema import Risk

SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    tool       TEXT NOT NULL,
    scope      TEXT NOT NULL,
    risk       INTEGER NOT NULL,
    granted_at TEXT NOT NULL,
    PRIMARY KEY (tool, scope)
);
"""


@dataclass(frozen=True)
class Approval:
    """One standing permission."""

    tool: str
    scope: str
    risk: Risk
    granted_at: str


class ApprovalStore:
    """Remembers what the user has already said yes to."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        # A single shared connection: ":memory:" would otherwise be a new,
        # empty database on every connect.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def is_approved(self, tool: str, scope: str, risk: Risk) -> bool:
        """True when a standing approval covers at least this much risk.

        An approval granted for a write does not cover a later call that turns
        out to be destructive, even on the same scope.
        """
        with closing(self._connection.execute(
            "SELECT risk FROM approvals WHERE tool = ? AND scope = ?", (tool, scope)
        )) as cursor:
            row = cursor.fetchone()
        return row is not None and int(row["risk"]) >= int(risk)

    def approve(self, tool: str, scope: str, risk: Risk) -> Approval:
        """Record an approval, widening an existing one if need be."""
        granted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO approvals (tool, scope, risk, granted_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(tool, scope) DO UPDATE SET "
                    "  risk = MAX(risk, excluded.risk), granted_at = excluded.granted_at",
                    (tool, scope, int(risk), granted_at),
                )
        except sqlite3.Error as e:
            # Worth saying out loud: the user will be asked again next time.
            logger.error(f"Could not store approval for {tool}/{scope}: {e}")
        logger.info(f"Approved {tool} for {scope} at {risk.name}")
        return Approval(tool, scope, risk, granted_at)

    def revoke(self, tool: str, scope: str) -> bool:
        """Withdraw one approval. Returns whether anything was removed."""
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM approvals WHERE tool = ? AND scope = ?", (tool, scope)
            )
        return cursor.rowcount > 0

    def get(self, tool: str, scope: str) -> Approval | None:
        """The stored approval for a tool and scope, if any."""
        with closing(self._connection.execute(
            "SELECT * FROM approvals WHERE tool = ? AND scope = ?", (tool, scope)
        )) as cursor:
            row = cursor.fetchone()
        return self._to_approval(row) if row else None

    def all(self) -> list[Approval]:
        """Every approval, newest first - what the user would want to review."""
        with closing(self._connection.execute(
            "SELECT * FROM approvals ORDER BY granted_at DESC, tool, scope"
        )) as cursor:
            return [self._to_approval(row) for row in cursor.fetchall()]

    def clear(self) -> int:
        """Forget every approval."""
        with self._connection:
            cursor = self._connection.execute("DELETE FROM approvals")
        return cursor.rowcount

    @staticmethod
    def _to_approval(row: sqlite3.Row) -> Approval:
        return Approval(
            tool=row["tool"],
            scope=row["scope"],
            risk=Risk(int(row["risk"])),
            granted_at=row["granted_at"],
        )
