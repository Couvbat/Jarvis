"""Filesystem tools: the full CRUD surface, sandboxed.

Split by concern rather than by operation. One tool with an ``operation`` enum
keeps the prompt small, but its parameter schema becomes a union in which
nothing says which argument belongs to which operation - and that is exactly
the kind of ambiguity a small local model gets wrong. Each tool here has a
short, unambiguous schema instead.

These functions are pure capability: they validate the path and do the work.
Deciding whether a call may happen at all is the policy engine's job.
"""

from __future__ import annotations

import functools
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

from policy.paths import PathPolicy
from tools.schema import Risk, ToolResult, ToolSpec, namespaced

NAMESPACE = "fs"

#: A local model's context is small; never hand it a whole large file.
DEFAULT_MAX_LINES = 200
MAX_LINE_LENGTH = 1000
MAX_LIST_ENTRIES = 200
MAX_SEARCH_RESULTS = 50
#: Stop walking rather than hang on a huge tree.
MAX_SEARCH_SCAN = 5000
#: Bytes inspected when deciding whether a file is text.
BINARY_SNIFF_BYTES = 8192


# --------------------------------------------------------------------------- #
# Argument coercion
#
# Arguments come from a language model, which routinely emits "10" where an
# integer is wanted and "true" where a boolean is. Accepting those costs
# nothing; rejecting genuinely unusable values still reports a clear error.
# --------------------------------------------------------------------------- #

def _as_int(value: Any, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got a boolean")
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number, got {value!r}") from e


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "yes", "y", "1", "on")


def _looks_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return b"\x00" in handle.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False


def guarded(method: Callable) -> Callable:
    """Turn a refused path or an unusable argument into a result, not a traceback.

    The model needs to read the reason and correct itself, so these are
    ordinary results rather than exceptions bubbling up through the registry.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except PermissionError as e:
            return ToolResult.error(str(e))
        except ValueError as e:
            return ToolResult.error(str(e))

    return wrapper


class FilesystemTools:
    """CRUD over the sandbox described by a :class:`PathPolicy`."""

    def __init__(self, path_policy: PathPolicy):
        self.paths = path_policy

    # -- helpers ---------------------------------------------------------- #

    def _resolve(self, raw_path: Any) -> Path:
        """Resolve and validate, or raise PermissionError with the reason."""
        if raw_path is None or str(raw_path).strip() == "":
            raise ValueError("path is required")
        verdict = self.paths.check(raw_path)
        if not verdict.allowed:
            raise PermissionError(verdict.reason)
        return verdict.path

    # -- read ------------------------------------------------------------- #

    @guarded
    def read(self, path: Any, start_line: Any = None, max_lines: Any = None) -> ToolResult:
        """Read a text file, optionally a slice of it."""
        target = self._resolve(path)
        start = max(1, _as_int(start_line, 1, "start_line"))
        limit = max(1, _as_int(max_lines, DEFAULT_MAX_LINES, "max_lines"))

        if not target.exists():
            return ToolResult.error(f"no such file: {target}")
        if not target.is_file():
            return ToolResult.error(f"not a file: {target}")
        if _looks_binary(target):
            return ToolResult.error(f"{target} looks like a binary file")

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return ToolResult.error(f"could not read {target}: {e}")

        window = lines[start - 1:start - 1 + limit]
        clipped = [
            line if len(line) <= MAX_LINE_LENGTH else line[:MAX_LINE_LENGTH] + " …"
            for line in window
        ]

        header = f"{target} (lines {start}-{start + len(window) - 1} of {len(lines)})"
        if not window:
            header = f"{target} ({len(lines)} lines; nothing at line {start})"

        return ToolResult("\n".join([header, *clipped]))

    @guarded
    def list_directory(self, path: Any) -> ToolResult:
        """List the entries of a directory."""
        target = self._resolve(path)

        if not target.exists():
            return ToolResult.error(f"no such directory: {target}")
        if not target.is_dir():
            return ToolResult.error(f"not a directory: {target}")

        try:
            entries = sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name))
        except OSError as e:
            return ToolResult.error(f"could not list {target}: {e}")

        rendered = []
        for entry in entries[:MAX_LIST_ENTRIES]:
            if entry.is_dir():
                rendered.append(f"{entry.name}/")
            else:
                try:
                    rendered.append(f"{entry.name} ({entry.stat().st_size} bytes)")
                except OSError:
                    rendered.append(entry.name)

        if len(entries) > MAX_LIST_ENTRIES:
            rendered.append(f"… and {len(entries) - MAX_LIST_ENTRIES} more")

        body = "\n".join(rendered) if rendered else "(empty)"
        return ToolResult(f"{target}:\n{body}")

    @guarded
    def stat(self, path: Any) -> ToolResult:
        """Report a path's type, size and modification time."""
        target = self._resolve(path)

        if not target.exists():
            return ToolResult.error(f"no such path: {target}")

        try:
            info = target.stat()
        except OSError as e:
            return ToolResult.error(f"could not stat {target}: {e}")

        kind = "directory" if target.is_dir() else "file"
        modified = datetime.fromtimestamp(info.st_mtime).isoformat(timespec="seconds")
        return ToolResult(
            f"{target}\ntype: {kind}\nsize: {info.st_size} bytes\nmodified: {modified}"
        )

    @guarded
    def search(
        self,
        path: Any,
        name_pattern: Any = None,
        content_pattern: Any = None,
        max_results: Any = None,
    ) -> ToolResult:
        """Find files by name glob and/or by text content."""
        root = self._resolve(path)
        limit = max(1, _as_int(max_results, MAX_SEARCH_RESULTS, "max_results"))

        if not root.is_dir():
            return ToolResult.error(f"not a directory: {root}")
        if not name_pattern and not content_pattern:
            return ToolResult.error("give a name_pattern, a content_pattern, or both")

        needle = str(content_pattern).lower() if content_pattern else None
        matches: List[str] = []
        scanned = 0
        truncated = False

        for candidate in root.rglob(str(name_pattern) if name_pattern else "*"):
            scanned += 1
            if scanned > MAX_SEARCH_SCAN:
                truncated = True
                break
            if not candidate.is_file():
                continue
            # Denied names must not surface through search either.
            if not self.paths.check(candidate).allowed:
                continue

            if needle is None:
                matches.append(str(candidate))
            else:
                if _looks_binary(candidate):
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for number, line in enumerate(text.splitlines(), start=1):
                    if needle in line.lower():
                        matches.append(f"{candidate}:{number}: {line.strip()[:200]}")
                        break

            if len(matches) >= limit:
                truncated = True
                break

        if not matches:
            return ToolResult(f"No match under {root}")

        body = "\n".join(matches)
        if truncated:
            body += f"\n… stopped at {len(matches)} results"
        return ToolResult(body)

    # -- create and update ------------------------------------------------ #

    @guarded
    def write(self, path: Any, content: Any = "", mode: Any = "create") -> ToolResult:
        """Create, overwrite or append to a file."""
        target = self._resolve(path)
        mode = str(mode or "create").strip().lower()
        if mode not in ("create", "overwrite", "append"):
            return ToolResult.error(
                f"mode must be create, overwrite or append, got {mode!r}"
            )

        text = "" if content is None else str(content)

        if mode == "create" and target.exists():
            return ToolResult.error(
                f"{target} already exists; use mode=overwrite to replace it "
                f"or mode=append to add to it"
            )
        if mode == "append" and target.is_dir():
            return ToolResult.error(f"not a file: {target}")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with open(target, "a", encoding="utf-8") as handle:
                    handle.write(text)
            else:
                target.write_text(text, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"could not write {target}: {e}")

        verb = {"create": "Created", "overwrite": "Overwrote", "append": "Appended to"}[mode]
        return ToolResult(f"{verb} {target} ({len(text)} characters)")

    @guarded
    def edit(self, path: Any, old_text: Any, new_text: Any = "") -> ToolResult:
        """Replace one exact, unique piece of text inside a file.

        Uniqueness is required: an ambiguous match would otherwise be resolved
        arbitrarily, and the model would have no way to tell which occurrence
        it changed.
        """
        target = self._resolve(path)

        if not old_text:
            return ToolResult.error("old_text is required")
        old = str(old_text)
        new = "" if new_text is None else str(new_text)

        if not target.is_file():
            return ToolResult.error(f"not a file: {target}")
        if _looks_binary(target):
            return ToolResult.error(f"{target} looks like a binary file")

        try:
            original = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return ToolResult.error(f"could not read {target}: {e}")

        occurrences = original.count(old)
        if occurrences == 0:
            return ToolResult.error(f"old_text does not appear in {target}")
        if occurrences > 1:
            return ToolResult.error(
                f"old_text appears {occurrences} times in {target}; "
                f"include more surrounding text so it matches exactly once"
            )

        try:
            target.write_text(original.replace(old, new, 1), encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"could not write {target}: {e}")

        return ToolResult(f"Edited {target}")

    @guarded
    def make_directory(self, path: Any) -> ToolResult:
        """Create a directory, with its parents."""
        target = self._resolve(path)
        if target.is_file():
            return ToolResult.error(f"{target} exists and is a file")
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return ToolResult.error(f"could not create {target}: {e}")
        return ToolResult(f"Created directory {target}")

    # -- delete and relocate ---------------------------------------------- #

    @guarded
    def delete(self, path: Any, recursive: Any = None) -> ToolResult:
        """Delete a file, or a directory when ``recursive`` is set."""
        target = self._resolve(path)
        recurse = _as_bool(recursive, False)

        if not target.exists():
            return ToolResult.error(f"no such path: {target}")

        if target in self.paths.allowed_directories:
            return ToolResult.error(f"refusing to delete the sandbox root {target}")

        try:
            if target.is_dir():
                if not recurse:
                    if any(target.iterdir()):
                        return ToolResult.error(
                            f"{target} is a non-empty directory; "
                            f"pass recursive=true to delete it and its contents"
                        )
                    target.rmdir()
                    return ToolResult(f"Deleted empty directory {target}")
                count = sum(1 for _ in target.rglob("*"))
                shutil.rmtree(target)
                return ToolResult(f"Deleted directory {target} and {count} entries")
            target.unlink()
            return ToolResult(f"Deleted {target}")
        except OSError as e:
            return ToolResult.error(f"could not delete {target}: {e}")

    @guarded
    def move(self, source: Any, destination: Any) -> ToolResult:
        """Move or rename a file or directory. Both ends must be in the sandbox."""
        origin = self._resolve(source)
        target = self._resolve(destination)

        if not origin.exists():
            return ToolResult.error(f"no such path: {origin}")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(origin), str(target))
        except OSError as e:
            return ToolResult.error(f"could not move {origin}: {e}")
        return ToolResult(f"Moved {origin} to {target}")

    @guarded
    def copy(self, source: Any, destination: Any) -> ToolResult:
        """Copy a file or directory. Both ends must be in the sandbox."""
        origin = self._resolve(source)
        target = self._resolve(destination)

        if not origin.exists():
            return ToolResult.error(f"no such path: {origin}")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if origin.is_dir():
                shutil.copytree(origin, target, dirs_exist_ok=True)
            else:
                shutil.copy2(origin, target)
        except OSError as e:
            return ToolResult.error(f"could not copy {origin}: {e}")
        return ToolResult(f"Copied {origin} to {target}")


# --------------------------------------------------------------------------- #
# Argument-aware risk
# --------------------------------------------------------------------------- #

def _write_risk(arguments: dict) -> Risk:
    """Overwriting something that exists destroys it; creating does not."""
    mode = str(arguments.get("mode") or "create").strip().lower()
    if mode != "overwrite":
        return Risk.WRITE
    path = arguments.get("path")
    return Risk.DESTRUCTIVE if path and Path(str(path)).expanduser().exists() else Risk.WRITE


def _clobbers_destination(arguments: dict) -> Risk:
    destination = arguments.get("destination")
    if destination and Path(str(destination)).expanduser().exists():
        return Risk.DESTRUCTIVE
    return Risk.WRITE


def _parent_scope(arguments: dict) -> str:
    """Approvals for a file operation cover the directory it happens in."""
    path = arguments.get("path")
    return str(Path(str(path or ".")).expanduser().parent)


def _directory_scope(arguments: dict) -> str:
    """For tools whose argument is itself a directory."""
    path = arguments.get("path")
    return str(Path(str(path or ".")).expanduser())


def _relocation_scope(arguments: dict) -> str:
    """Moving or copying touches two directories; name both."""
    source = Path(str(arguments.get("source") or ".")).expanduser().parent
    destination = Path(str(arguments.get("destination") or ".")).expanduser().parent
    return f"{source} -> {destination}"


def _string(description: str) -> dict:
    return {"type": "string", "description": description}


def _path_precheck(path_policy: PathPolicy, *keys: str):
    """Refuse a call whose paths are out of bounds, before anyone is asked."""
    def check(arguments: dict) -> Optional[str]:
        for key in keys:
            value = arguments.get(key)
            if value is None or str(value).strip() == "":
                continue
            verdict = path_policy.check(value)
            if not verdict.allowed:
                return verdict.reason
        return None

    return check


def build_tools(path_policy: PathPolicy) -> List[ToolSpec]:
    """Build every filesystem tool against one sandbox."""
    fs = FilesystemTools(path_policy)
    check_path = _path_precheck(path_policy, "path")
    check_relocation = _path_precheck(path_policy, "source", "destination")

    def spec(name, handler, description, properties, required, risk,
             risk_for=None, scope_for=_parent_scope, precheck=None):
        return ToolSpec(
            name=namespaced(NAMESPACE, name),
            description=description,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            handler=handler,
            risk=risk,
            risk_for=risk_for,
            scope_for=scope_for,
            precheck=precheck if precheck is not None else check_path,
        )

    return [
        spec(
            "read", fs.read,
            "Read the contents of a text file, optionally a range of lines",
            {
                "path": _string("Path of the file to read"),
                "start_line": {"type": "integer", "description": "First line (1-based)"},
                "max_lines": {"type": "integer", "description": "How many lines to return"},
            },
            ["path"], Risk.READ_ONLY,
        ),
        spec(
            "list", fs.list_directory,
            "List the files and directories inside a directory",
            {"path": _string("Directory to list")},
            ["path"], Risk.READ_ONLY, scope_for=_directory_scope,
        ),
        spec(
            "stat", fs.stat,
            "Report whether a path is a file or directory, its size and when it changed",
            {"path": _string("Path to inspect")},
            ["path"], Risk.READ_ONLY,
        ),
        spec(
            "search", fs.search,
            "Find files under a directory by name pattern and/or text content",
            {
                "path": _string("Directory to search under"),
                "name_pattern": _string("Glob for the file name, e.g. '*.py'"),
                "content_pattern": _string("Text to look for inside files"),
                "max_results": {"type": "integer", "description": "Maximum matches"},
            },
            ["path"], Risk.READ_ONLY, scope_for=_directory_scope,
        ),
        spec(
            "write", fs.write,
            "Write a file: create a new one, overwrite an existing one, or append",
            {
                "path": _string("Path of the file to write"),
                "content": _string("Text to write"),
                "mode": {
                    "type": "string",
                    "enum": ["create", "overwrite", "append"],
                    "description": "create fails if the file exists; overwrite replaces it",
                },
            },
            ["path", "content"], Risk.WRITE, _write_risk,
        ),
        spec(
            "edit", fs.edit,
            "Replace one exact, unique piece of text inside an existing file",
            {
                "path": _string("Path of the file to edit"),
                "old_text": _string("Exact text to replace; must appear exactly once"),
                "new_text": _string("Replacement text"),
            },
            ["path", "old_text", "new_text"], Risk.WRITE,
        ),
        spec(
            "mkdir", fs.make_directory,
            "Create a directory, including any missing parent directories",
            {"path": _string("Directory to create")},
            ["path"], Risk.WRITE, scope_for=_directory_scope,
        ),
        spec(
            "delete", fs.delete,
            "Delete a file, or a directory and its contents when recursive is true",
            {
                "path": _string("Path to delete"),
                "recursive": {
                    "type": "boolean",
                    "description": "Required to delete a non-empty directory",
                },
            },
            ["path"], Risk.DESTRUCTIVE,
        ),
        spec(
            "move", fs.move,
            "Move or rename a file or directory",
            {
                "source": _string("Path to move"),
                "destination": _string("Where to move it to"),
            },
            ["source", "destination"], Risk.WRITE, _clobbers_destination,
            scope_for=_relocation_scope, precheck=check_relocation,
        ),
        spec(
            "copy", fs.copy,
            "Copy a file or directory",
            {
                "source": _string("Path to copy"),
                "destination": _string("Where to copy it to"),
            },
            ["source", "destination"], Risk.WRITE, _clobbers_destination,
            scope_for=_relocation_scope, precheck=check_relocation,
        ),
    ]
