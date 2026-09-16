"""Connecting to MCP servers and keeping them isolated from each other.

Each server gets its own session group *and its own task*. The task both
enters and leaves the connection's async context, which is what anyio
requires: entering a cancel scope in one task and exiting it in another is an
error, and gathering several connects onto one shared exit stack would do
exactly that.

The isolation is also the point operationally. A server that will not start,
dies mid-session or stops answering must cost the user that server and
nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
from enum import Enum
from typing import Any

from loguru import logger
from mcp import StdioServerParameters
from mcp.client.session_group import (
    ClientSessionGroup,
    SseServerParameters,
    StreamableHttpParameters,
)

from tools.mcp.servers import ServerConfig, Transport
from tools.schema import NAMESPACE_SEPARATOR, ToolResult

#: Seconds to wait for a server to connect and list its tools.
DEFAULT_CONNECT_TIMEOUT = 15.0
#: Seconds to wait for one tool call.
DEFAULT_CALL_TIMEOUT = 60.0
#: Seconds to wait for a server to shut down before giving up on it.
SHUTDOWN_TIMEOUT = 5.0


class ServerStatus(str, Enum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    DISABLED = "disabled"


def result_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into text for the model."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
            continue
        kind = getattr(block, "type", type(block).__name__)
        parts.append(f"[{kind} content omitted]")

    if not parts:
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return str(structured)
    return "\n".join(parts)


class McpServer:
    """One MCP server connection, owned end to end by a single task."""

    def __init__(
        self,
        config: ServerConfig,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
    ):
        self.config = config
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout

        self.status = ServerStatus.DISABLED if not config.enabled else ServerStatus.STOPPED
        self.error = ""
        self.tools: dict[str, Any] = {}

        self._group: ClientSessionGroup | None = None
        self._task: asyncio.Task | None = None
        self._was_connected = False
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()

    # -- naming ----------------------------------------------------------- #

    @property
    def namespace(self) -> str:
        return self.config.name

    @property
    def is_connected(self) -> bool:
        return self.status is ServerStatus.CONNECTED and self._group is not None

    def _qualify(self, tool_name: str, server_info: Any = None) -> str:
        return f"{self.namespace}{NAMESPACE_SEPARATOR}{tool_name}"

    # -- connection parameters -------------------------------------------- #

    def _parameters(self):
        config = self.config
        if config.transport is Transport.STDIO:
            return StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env=dict(config.env) or None,
                cwd=config.cwd,
            )
        if config.transport is Transport.SSE:
            return SseServerParameters(
                url=config.url, headers=dict(config.headers), timeout=config.timeout
            )
        return StreamableHttpParameters(
            url=config.url, headers=dict(config.headers), timeout=config.timeout
        )

    # -- lifecycle -------------------------------------------------------- #

    async def start(self) -> bool:
        """Connect and list tools. Returns whether the server is usable."""
        if not self.config.enabled:
            logger.info(f"MCP server '{self.namespace}' is disabled")
            return False

        self.status = ServerStatus.CONNECTING
        self._ready.clear()
        self._shutdown.clear()
        self._task = asyncio.create_task(self._run(), name=f"mcp:{self.namespace}")

        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self.connect_timeout)
        except asyncio.TimeoutError:
            self.status = ServerStatus.FAILED
            self.error = f"did not connect within {self.connect_timeout:.0f}s"
            logger.error(f"MCP server '{self.namespace}' {self.error}")
            await self.stop()
            return False

        return self.status is ServerStatus.CONNECTED

    async def _run(self) -> None:
        """Own the connection for its whole life, in one task."""
        try:
            async with contextlib.AsyncExitStack() as stack:
                group = ClientSessionGroup(
                    exit_stack=stack, component_name_hook=self._qualify
                )
                await group.connect_to_server(self._parameters())

                self._group = group
                self.tools = dict(group.tools)
                self.status = ServerStatus.CONNECTED
                self._was_connected = True
                self.error = ""
                logger.info(
                    f"MCP server '{self.namespace}' connected "
                    f"with {len(self.tools)} tool(s)"
                )
                self._ready.set()

                await self._shutdown.wait()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.status = ServerStatus.FAILED
            self.error = f"{type(e).__name__}: {e}"
            logger.error(f"MCP server '{self.namespace}' failed: {self.error}")
        finally:
            self._group = None
            self.tools = {}
            # Unblocks start() on the failure path as well as the happy one.
            self._ready.set()

    async def attach_session(self, server_info: Any, session: Any) -> bool:
        """Attach an already-connected session, for tests and embedded servers."""
        stack = contextlib.AsyncExitStack()
        group = ClientSessionGroup(exit_stack=stack, component_name_hook=self._qualify)
        await group.connect_with_session(server_info, session)
        self._group = group
        self.tools = dict(group.tools)
        self.status = ServerStatus.CONNECTED
        self._was_connected = True
        self._ready.set()
        return True

    async def stop(self) -> None:
        """Disconnect, and do not let a hung server hold up shutdown."""
        self._shutdown.set()
        task, self._task = self._task, None

        if task is not None and not task.done():
            if self._was_connected:
                # A live session gets a moment to close cleanly.
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=SHUTDOWN_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"MCP server '{self.namespace}' did not stop; cancelling"
                    )
                    await self._cancel(task)
                except Exception as e:  # pragma: no cover - _run already reported
                    logger.debug(f"MCP server '{self.namespace}' stopped with {e}")
            else:
                # It never connected, so it is stuck somewhere in startup and
                # there is nothing to close gracefully. Waiting out the grace
                # period here just delays every other server behind it.
                await self._cancel(task)

        self._group = None
        self.tools = {}
        self._was_connected = False
        if self.status is not ServerStatus.FAILED:
            self.status = ServerStatus.STOPPED

    @staticmethod
    async def _cancel(task: asyncio.Task) -> None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # -- calling ---------------------------------------------------------- #

    async def call(self, qualified_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run one tool on this server."""
        if not self.is_connected:
            return ToolResult.error(
                f"MCP server '{self.namespace}' is not connected ({self.status.value})"
            )

        try:
            raw = await asyncio.wait_for(
                self._group.call_tool(qualified_name, arguments),
                timeout=self.call_timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult.error(
                f"'{qualified_name}' timed out after {self.call_timeout:.0f}s"
            )
        except Exception as e:
            # A server can die mid-call; that is its problem, not the session's.
            logger.error(f"MCP call {qualified_name} failed: {e}")
            return ToolResult.error(f"'{qualified_name}' failed: {e}")

        text = result_text(raw)
        if getattr(raw, "is_error", False):
            return ToolResult(
                content=f"Error: {text}", ok=False,
                untrusted=True, origin=self.namespace,
            )
        # Everything a third-party server returns is content the user did not
        # write, so it taints the turn exactly like a fetched web page. The
        # origin is the server, so reading more from it is not treated as
        # carrying its content somewhere new.
        return ToolResult(content=text, untrusted=True, origin=self.namespace)


class McpManager:
    """Every configured server, started in parallel and isolated from each other."""

    def __init__(
        self,
        configs: list[ServerConfig],
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
    ):
        self.servers: list[McpServer] = [
            McpServer(config, connect_timeout, call_timeout) for config in configs
        ]

    def __len__(self) -> int:
        return len(self.servers)

    @property
    def connected(self) -> list[McpServer]:
        return [s for s in self.servers if s.status is ServerStatus.CONNECTED]

    def statuses(self) -> dict[str, str]:
        """What happened to each server, for the UI and the logs."""
        return {
            server.namespace: (
                server.status.value if not server.error
                else f"{server.status.value}: {server.error}"
            )
            for server in self.servers
        }

    def server_for(self, qualified_name: str) -> McpServer | None:
        """Which server owns a namespaced tool name."""
        namespace = qualified_name.split(NAMESPACE_SEPARATOR, 1)[0]
        for server in self.servers:
            if server.namespace == namespace:
                return server
        return None

    async def start(self) -> None:
        """Start every enabled server. One failure never stops the others."""
        if not self.servers:
            return
        results = await asyncio.gather(
            *(server.start() for server in self.servers), return_exceptions=True
        )
        for server, result in zip(self.servers, results, strict=True):
            if isinstance(result, BaseException):  # pragma: no cover - defensive
                server.status = ServerStatus.FAILED
                server.error = str(result)
        logger.info(f"MCP servers: {self.statuses()}")

    async def stop(self) -> None:
        """Disconnect everything."""
        await asyncio.gather(
            *(server.stop() for server in self.servers), return_exceptions=True
        )
