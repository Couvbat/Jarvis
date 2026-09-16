"""Launching applications and whitelisted commands.

The command whitelist is a *separate, weaker* gate than the filesystem
sandbox: a whitelisted `rm` or `cat` reaches the whole disk, because it never
goes through PathPolicy. So the default whitelist holds application launchers
only, and anything that manipulates files is expected to go through the
sandboxed ``fs__*`` tools instead.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from tools.schema import Risk, ToolResult, ToolSpec, namespaced

NAMESPACE = "app"

CLI_TIMEOUT_SECONDS = 5
MAX_OUTPUT_CHARS = 2000


class AppTools:
    """Runs commands, but only the ones the user listed."""

    def __init__(
        self,
        command_whitelist: Sequence[str],
        gui_applications: Sequence[str] = (),
        timeout: int = CLI_TIMEOUT_SECONDS,
    ):
        self.command_whitelist = [c.strip() for c in command_whitelist if c and c.strip()]
        self.gui_applications = [g.strip() for g in gui_applications if g and g.strip()]
        self.timeout = timeout

    def _resolve_executable(self, executable: str) -> tuple[str | None, str]:
        """Resolve a command to a path, refusing anything PATH would not give.

        Comparing raw strings is not enough: "/bin/ls" and "ls" name the same
        program but are different strings, and "/tmp/evil/ls" has the right
        basename while being something else entirely. So the basename must be
        whitelisted *and* the binary must be the one PATH resolves to.
        """
        if os.sep in executable:
            if not os.path.isabs(executable):
                return None, f"refusing a relative command path: {executable}"
            given = Path(executable)
            expected = shutil.which(given.name)
            if expected is None:
                return None, f"command not found on PATH: {given.name}"
            try:
                same = Path(expected).resolve() == given.resolve()
            except OSError as e:
                return None, f"could not resolve {executable}: {e}"
            if not same:
                return None, (
                    f"{executable} is not the '{given.name}' on PATH ({expected})"
                )
            return str(given), ""

        found = shutil.which(executable)
        if found is None:
            return None, f"command not found: {executable}"
        return found, ""

    def launch(self, application: Any, args: Any = None) -> ToolResult:
        """Launch an application or run a whitelisted command."""
        if not application or not str(application).strip():
            return ToolResult.error("application is required")

        try:
            # Split here as well as at the exec, so "ls -la" as a single string
            # is understood rather than looked up as a binary of that name.
            parts = shlex.split(str(application))
        except ValueError as e:
            return ToolResult.error(f"could not parse command: {e}")
        if not parts:
            return ToolResult.error("application is required")

        executable, inline_args = parts[0], parts[1:]

        extra: list[str] = []
        if args:
            extra = [str(a) for a in args] if isinstance(args, (list, tuple)) else [str(args)]

        name = Path(executable).name
        if name not in self.command_whitelist:
            return ToolResult.error(
                f"'{name}' is not in the command whitelist "
                f"({', '.join(self.command_whitelist) or 'empty'})"
            )

        resolved, reason = self._resolve_executable(executable)
        if resolved is None:
            return ToolResult.error(reason)

        command = [resolved, *inline_args, *extra]
        logger.info(f"Running {command}")

        try:
            if name in self.gui_applications:
                subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return ToolResult(f"Launched {name}")

            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error(f"'{name}' timed out after {self.timeout}s")
        except OSError as e:
            return ToolResult.error(f"could not run '{name}': {e}")

        output = (completed.stdout or "").strip()
        if completed.stderr:
            output = f"{output}\nErrors: {completed.stderr.strip()}".strip()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n… (truncated)"

        if completed.returncode != 0:
            return ToolResult.error(
                f"'{name}' exited with status {completed.returncode}\n{output}".strip()
            )
        return ToolResult(output or f"'{name}' finished with no output")


def _whitelist_precheck(command_whitelist: Sequence[str]):
    """Refuse a command that is not whitelisted, before anyone is asked."""
    allowed = [c.strip() for c in command_whitelist if c and c.strip()]

    def check(arguments: dict) -> str | None:
        raw = str(arguments.get("application") or "").strip()
        if not raw:
            return "application is required"
        try:
            parts = shlex.split(raw)
        except ValueError as e:
            return f"could not parse command: {e}"
        if not parts:
            return "application is required"
        name = Path(parts[0]).name
        if name not in allowed:
            return (
                f"'{name}' is not in the command whitelist "
                f"({', '.join(allowed) or 'empty'})"
            )
        return None

    return check


def _command_scope(arguments: dict) -> str:
    """Approvals for a launch cover that exact command line and nothing else."""
    application = str(arguments.get("application") or "")
    args = arguments.get("args") or []
    if isinstance(args, (list, tuple)):
        rendered = " ".join(str(a) for a in args)
    else:
        rendered = str(args)
    return f"{application} {rendered}".strip()


def build_tools(
    command_whitelist: Sequence[str],
    gui_applications: Sequence[str] = (),
    timeout: int = CLI_TIMEOUT_SECONDS,
) -> list[ToolSpec]:
    """Build the application tools."""
    apps = AppTools(command_whitelist, gui_applications, timeout)
    return [
        ToolSpec(
            name=namespaced(NAMESPACE, "launch"),
            description=(
                "Launch an application or run a command from the user's whitelist. "
                "Use the fs__ tools for file operations, not shell commands."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "string",
                        "description": "Name of the application or command to run",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments to pass to it",
                    },
                },
                "required": ["application"],
            },
            handler=apps.launch,
            scope_for=_command_scope,
            precheck=_whitelist_precheck(command_whitelist),
            # Whatever the whitelist allows, running it is outside the
            # filesystem sandbox, so this never counts as a safe call.
            risk=Risk.DESTRUCTIVE,
        ),
    ]
