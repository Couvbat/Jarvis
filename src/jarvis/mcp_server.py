"""Jarvis the other way round: its tools, offered over MCP.

Jarvis is an MCP *client* - that is Phase 2 and the point of the project. This
is the mirror: the same sandboxed filesystem tools, the same document search,
the same reminders, exposed to any MCP client. A self-hosted setup then has
one place where "which directories may be touched" is decided, rather than
one per client.

The whole design turns on one question: **who answers the confirmation?**

Interactively a person does, and the policy engine is built around that:
even a plain read gets a confirmation, because it is cheap to say yes once
and remember it. Here the caller is a program and there is nobody to ask, so
deferring to the engine unchanged would refuse everything, and auto-approving
would discard the only thing standing between a model and the disk.

So the rule is stated here, explicitly, rather than pretended away:

- **Refused always**: anything the sandbox rejects, anything destructive,
  and anything the engine escalates to the keyboard - which includes every
  call made after untrusted content entered this session.
- **Refused here**: egress. Fetching a page is read-only in risk terms and
  still a request leaving the user's machine, which is a decision for the
  person whose machine it is.
- **Allowed**: reads inside the configured sandbox, and whatever the user
  approved out of band in Jarvis itself. The sandbox *is* the authorisation
  for a read - it is the setting whose whole job is saying which files a
  program may see.

That is looser than an interactive session for reads and stricter for
everything else, which is the honest trade for having no one to ask. The
listing holds only what could succeed, because a toolbox full of things that
cannot be called is worse than a small one; the rule below is the gate, and a
client that knows a name can still call it and be told no.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

import mcp.types as types
from loguru import logger
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from jarvis.config import settings
from jarvis.policy.engine import PolicyEngine, Surface
from jarvis.policy.store import ApprovalStore
from jarvis.policy.taint import TaintState
from jarvis.tools.builtin import build_default_registry
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.schema import Risk, ToolResult, ToolSpec

SERVER_NAME = "jarvis"

INSTRUCTIONS = """Jarvis's own tools: sandboxed filesystem access, web
fetching, document search over the user's indexed notes, and reminders.

Only calls that need no human confirmation are available. Anything
destructive, or anything reached after untrusted content entered the
conversation, is refused with a reason rather than silently approved -
there is nobody at a keyboard here to ask."""


def offerable(spec: ToolSpec) -> bool:
    """Whether this tool could succeed with nobody to ask.

    Baseline risk, not the per-call decision: the engine still runs on every
    call. This only keeps the listing honest.
    """
    if spec.preapproved:
        return True
    # Egress is read-only in risk terms and still a request leaving the
    # user's machine, which the engine asks about. Advertising it here would
    # be advertising a refusal.
    return spec.risk <= Risk.READ_ONLY and not spec.egress


def permitted(spec: ToolSpec, decision) -> bool:
    """Whether this call may run with nobody to confirm it.

    Kept apart from the listing and from the transport so the rule can be
    read, and tested, in one place.
    """
    if decision.surface is Surface.NONE:
        # Preapproved, or a standing approval the user granted in Jarvis.
        return True
    if decision.surface is Surface.TERMINAL:
        # Destructive, or made after untrusted content entered the session.
        # These are the calls a person should see every time, so there is
        # nothing to fall back on.
        return False
    return spec.risk <= Risk.READ_ONLY and not spec.egress


def to_mcp_tool(spec: ToolSpec) -> types.Tool:
    """One Jarvis tool as the protocol describes it."""
    return types.Tool(
        name=spec.name,
        description=spec.description,
        input_schema=spec.input_schema,
        annotations=types.ToolAnnotations(
            read_only_hint=spec.risk <= Risk.READ_ONLY,
            destructive_hint=spec.risk >= Risk.DESTRUCTIVE,
            open_world_hint=spec.egress,
        ),
    )


class JarvisMcpServer:
    """Serves Jarvis's registry over MCP."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        policy: PolicyEngine | None = None,
    ):
        self.registry = registry if registry is not None else build_default_registry()
        self.policy = policy if policy is not None else PolicyEngine(
            ApprovalStore(settings.approvals_path)
        )
        #: One conversation's worth of taint. A client that reads a web page
        #: and then asks to write a file gets the same escalation a spoken
        #: session would - which here means a refusal.
        self.taint = TaintState()

    # -- protocol ----------------------------------------------------------- #

    def tools(self) -> list[types.Tool]:
        return [
            to_mcp_tool(spec) for spec in self.registry.specs() if offerable(spec)
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run one call, through the same policy engine as a spoken turn."""
        spec = self.registry.get(name)
        if spec is None:
            return ToolResult.error(f"no such tool: {name}")

        decision = self.policy.evaluate(spec, arguments or {}, self.taint)
        if not decision.allowed:
            return ToolResult.error(decision.reason)

        if not permitted(spec, decision):
            # The reason is passed on, so the caller can tell its user what
            # to do rather than leaving them guessing.
            return ToolResult.error(
                f"'{name}' needs a person to confirm it ({decision.reason}), "
                f"and this is an MCP server with nobody at the keyboard. "
                f"Ask the user to run it in Jarvis directly."
            )

        result = await self.registry.call(name, arguments or {})
        self.taint.observe(name, result)
        return result

    # -- wiring -------------------------------------------------------------- #

    def build(self) -> Server:
        """The MCP server object, handlers attached."""

        async def on_list_tools(context, params) -> types.ListToolsResult:
            return types.ListToolsResult(tools=self.tools())

        async def on_call_tool(context, params) -> types.CallToolResult:
            result = await self.call(params.name, dict(params.arguments or {}))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result.content)],
                is_error=not result.ok,
            )

        return Server(
            SERVER_NAME,
            version=_version(),
            instructions=INSTRUCTIONS,
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )

    async def serve_stdio(self) -> None:
        """Run over stdin/stdout, the way MCP clients launch a server."""
        server = self.build()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )


def _version() -> str:
    from jarvis import __version__

    return __version__


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jarvis-mcp",
        description=(
            "Serve Jarvis's tools over MCP on stdin/stdout. Only calls that "
            "need no human confirmation are available: there is nobody at a "
            "keyboard here to answer one."
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print the tools that would be offered, and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """The `jarvis-mcp` command."""
    arguments = parse_arguments(argv)

    # stdout is the protocol channel. A stray log line on it corrupts the
    # stream and the client sees a parse error rather than a message.
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)

    instance = JarvisMcpServer()

    if arguments.list:
        for tool in instance.tools():
            print(f"{tool.name}\t{tool.description.splitlines()[0]}")
        return 0

    logger.info(f"Serving {len(instance.tools())} tool(s) over MCP")
    try:
        asyncio.run(instance.serve_stdio())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
