"""Turning discovered MCP tools into registry entries.

The interesting question here is how much to believe a server about its own
tools. MCP tools carry annotations - ``read_only_hint``, ``destructive_hint``
and so on - but they are written by the server, which is third-party code. So
they are only ever allowed to make a tool look *more* dangerous. What makes a
tool look safer is the trust level the **user** set for that server.
"""

from __future__ import annotations

from typing import Any, List, Optional

from loguru import logger

from tools.mcp.servers import Trust
from tools.schema import Risk, ToolResult, ToolSpec, from_mcp_tool


def _hint(tool: Any, name: str) -> bool:
    annotations = getattr(tool, "annotations", None)
    return getattr(annotations, name, None) is True


def risk_for(tool: Any, trust: Trust) -> Optional[Risk]:
    """The risk to assign a tool, or None to not expose it at all."""
    read_only = _hint(tool, "read_only_hint")

    # A readonly server exposes only what it marks read-only. An unannotated
    # tool says nothing about itself, so it does not qualify.
    if trust is Trust.READONLY and not read_only:
        return None

    # Hardening always applies, whatever the trust level.
    if _hint(tool, "destructive_hint"):
        return Risk.DESTRUCTIVE

    if trust is Trust.TRUSTED:
        return Risk.READ_ONLY if read_only else Risk.WRITE

    if trust is Trust.READONLY:
        return Risk.READ_ONLY

    # CONFIRM, the default: every call is confirmed, and the user may approve
    # a tool so they are not asked again. read_only_hint buys nothing here -
    # relaxation comes from trust, not from the server's own claim.
    return Risk.WRITE


def build_specs(server: Any) -> List[ToolSpec]:
    """Registry entries for every tool a connected server exposes."""
    trust = server.config.trust
    specs: List[ToolSpec] = []

    for qualified_name, tool in server.tools.items():
        risk = risk_for(tool, trust)
        if risk is None:
            logger.info(
                f"Not exposing {qualified_name}: "
                f"'{server.namespace}' is readonly and the tool is not marked read-only"
            )
            continue

        # Marking a server "trusted" in the configuration *is* the approval
        # for its read-only tools; asking again on first use would be the
        # prompt fatigue this design keeps trying to avoid.
        preapproved = trust is Trust.TRUSTED and risk is Risk.READ_ONLY

        specs.append(from_mcp_tool(
            tool,
            server.namespace,
            handler=_make_handler(server, qualified_name),
            risk=risk,
            preapproved=preapproved,
            # There is no way to know what an MCP tool's arguments mean, so the
            # only meaningful unit to approve is the tool itself. Scoping by
            # exact arguments would almost never match twice - a search with a
            # different query is a different scope - and the approval would be
            # useless while the prompting continued.
            scope_for=_make_scope(qualified_name),
            origin_for=_make_origin(server.namespace),
            precheck=_make_precheck(server),
            # The session group already assigned this name and the handler
            # calls it; deriving it a second time would let the two drift.
            qualified_name=qualified_name,
        ))

    return specs


def _make_handler(server: Any, qualified_name: str):
    async def handler(**arguments: Any) -> ToolResult:
        return await server.call(qualified_name, arguments)

    handler.__name__ = f"call_{qualified_name}"
    return handler


def _make_scope(qualified_name: str):
    def scope(_arguments: dict) -> str:
        return qualified_name

    return scope


def _make_origin(namespace: str):
    def origin(_arguments: dict) -> str:
        return namespace

    return origin


def _make_precheck(server: Any):
    def precheck(_arguments: dict) -> Optional[str]:
        if not server.is_connected:
            return (
                f"MCP server '{server.namespace}' is not connected "
                f"({server.status.value})"
            )
        return None

    return precheck
