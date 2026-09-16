"""Where file operations are allowed to reach.

Two independent gates, in order:

1. **Sandbox** - the resolved path must sit inside one of the allowed
   directories. Resolution follows symlinks, so a link inside the sandbox
   pointing out of it does not get through.
2. **Denied patterns** - even inside an allowed directory, some names are off
   limits: SSH and cloud credentials, dotfiles holding secrets, shell history.
   These are matched only *below* the allowed root, so a user who deliberately
   allows a sensitive directory still gets what they asked for.

Both gates live here rather than in the filesystem tools, so that MCP
filesystem servers can be held to the same sandbox in Phase 2.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from loguru import logger

from config import settings


@dataclass(frozen=True)
class PathVerdict:
    """The outcome of checking one path."""

    allowed: bool
    path: Path
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class PathPolicy:
    """Decides whether a path may be touched at all."""

    def __init__(
        self,
        allowed_directories: Sequence[str | Path],
        denied_patterns: Iterable[str] = (),
    ):
        self.allowed_directories: list[Path] = []
        for directory in allowed_directories:
            candidate = Path(directory).expanduser()
            if not str(candidate).strip() or str(candidate) == ".":
                # An empty ALLOWED_DIRECTORIES entry would otherwise resolve to
                # the working directory, quietly allowing it.
                logger.warning("Ignoring empty entry in allowed directories")
                continue
            self.allowed_directories.append(candidate.resolve())

        self.denied_patterns = [p.strip() for p in denied_patterns if p and p.strip()]

    @classmethod
    def from_settings(cls) -> PathPolicy:
        """Build the policy the application is configured with."""
        return cls(settings.allowed_dirs_list, settings.denied_patterns_list)

    def describe_allowed(self) -> str:
        """Human-readable list of roots, for error messages and the UI."""
        return ", ".join(str(directory) for directory in self.allowed_directories) or "(none)"

    def _root_for(self, resolved: Path) -> Path | None:
        """The allowed directory containing this path, if any."""
        for directory in self.allowed_directories:
            if resolved == directory or directory in resolved.parents:
                return directory
        return None

    def _denied_component(self, resolved: Path, root: Path) -> str | None:
        """The first component below ``root`` that matches a denied pattern."""
        try:
            relative = resolved.relative_to(root)
        except ValueError:  # pragma: no cover - guarded by _root_for
            return None

        for component in relative.parts:
            for pattern in self.denied_patterns:
                if fnmatch(component, pattern):
                    return component
        return None

    def check(self, raw_path: str | Path) -> PathVerdict:
        """Check one path. The path need not exist: creating a file is normal."""
        try:
            candidate = Path(raw_path).expanduser()
        except (TypeError, ValueError, RuntimeError) as e:
            return PathVerdict(False, Path(str(raw_path)), f"invalid path: {e}")

        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError, ValueError) as e:
            # ValueError covers an embedded null byte, which a model-generated
            # argument can easily contain and which realpath refuses outright.
            return PathVerdict(False, candidate, f"could not resolve path: {e}")

        root = self._root_for(resolved)
        if root is None:
            return PathVerdict(
                False,
                resolved,
                f"'{raw_path}' is outside the allowed directories "
                f"({self.describe_allowed()})",
            )

        denied = self._denied_component(resolved, root)
        if denied is not None:
            return PathVerdict(
                False, resolved, f"'{denied}' is on the denied list"
            )

        return PathVerdict(True, resolved)
