"""The one place a tool call is turned into a function call.

Local tools and, from Phase 2, MCP tools register here side by side, so the
agent loop never needs to know where a capability comes from.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable
from typing import Any

from loguru import logger

from tools.schema import ToolResult, ToolSpec, to_ollama_schema


class DuplicateToolError(ValueError):
    """Raised when two providers claim the same qualified name."""


class ToolRegistry:
    """Holds the tool specs and dispatches calls to them."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        """Add a tool. Names are unique: silent shadowing would be a trapdoor."""
        if spec.name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        logger.debug(f"Registered tool: {spec.name} (risk={spec.risk.name})")
        return spec

    def register_all(self, specs: Iterable[ToolSpec]) -> None:
        """Add several tools."""
        for spec in specs:
            self.register(spec)

    def unregister_namespace(self, namespace: str) -> int:
        """Drop every tool from one provider, e.g. an MCP server going away."""
        prefix = f"{namespace}__"
        removed = [name for name in self._tools if name.startswith(prefix)]
        for name in removed:
            del self._tools[name]
        return len(removed)

    def get(self, name: str) -> ToolSpec | None:
        """The spec for a tool, or None if nothing is registered under it."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Every registered tool name, in registration order."""
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Every registered spec, in registration order."""
        return list(self._tools.values())

    def describe(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Function schemas for the model.

        ``names`` selects a subset, which is how Phase 2 will present only the
        tools relevant to a turn instead of all of them.
        """
        if names is None:
            selected = self.specs()
        else:
            selected = [self._tools[name] for name in names if name in self._tools]
        return [to_ollama_schema(spec) for spec in selected]

    async def call(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Run a tool and return its result.

        Handlers may be sync or async. Local ones are blocking I/O - files,
        subprocesses, HTTP - so they run in a worker thread rather than
        stalling the event loop; MCP handlers are already coroutines and are
        awaited directly.

        Never raises: a tool call comes from a language model, so a bad name,
        missing arguments or a handler blowing up are all routine and the model
        needs to see them as results it can react to.
        """
        spec = self.get(name)
        if spec is None:
            logger.warning(f"Unknown tool requested: {name}")
            return ToolResult.error(f"unknown tool '{name}'")

        arguments = dict(arguments or {})
        logger.info(f"Calling tool {name} with {arguments}")

        try:
            if inspect.iscoroutinefunction(spec.handler):
                result = await spec.handler(**arguments)
            else:
                result = await asyncio.to_thread(spec.handler, **arguments)
                if inspect.isawaitable(result):
                    result = await result
        except TypeError as e:
            # Wrong or missing arguments: tell the model what it got wrong.
            logger.warning(f"Bad arguments for {name}: {e}")
            return ToolResult.error(f"invalid arguments for '{name}': {e}")
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return ToolResult.error(f"'{name}' failed: {e}")

        if not isinstance(result, ToolResult):
            logger.error(f"Tool {name} returned {type(result).__name__}, not ToolResult")
            return ToolResult.error(f"'{name}' returned a malformed result")

        return result
