"""Assembling the tools Jarvis ships with.

One place that knows which local providers exist, so adding MCP servers in
Phase 2 is a matter of registering more specs into the same registry.
"""

from __future__ import annotations

from loguru import logger

from config import settings
from policy.paths import PathPolicy
from tools.local import apps, filesystem, web
from tools.mcp.adapter import build_specs
from tools.mcp.manager import McpManager
from tools.mcp.servers import load_servers
from tools.registry import DuplicateToolError, ToolRegistry


def build_default_registry(path_policy: PathPolicy | None = None) -> ToolRegistry:
    """Register every built-in tool, configured from settings."""
    paths = path_policy if path_policy is not None else PathPolicy.from_settings()

    registry = ToolRegistry()
    registry.register_all(filesystem.build_tools(paths))
    registry.register_all(web.build_tools(
        allow_private_network=settings.allow_private_network_fetch,
        max_bytes=settings.max_fetch_bytes,
        timeout=settings.fetch_timeout,
    ))
    registry.register_all(apps.build_tools(
        command_whitelist=settings.command_whitelist_list,
        gui_applications=settings.gui_applications_list,
    ))

    logger.info(
        f"Registered {len(registry.names())} tools; "
        f"sandbox: {paths.describe_allowed()}"
    )
    return registry


def build_mcp_manager() -> McpManager:
    """The configured MCP servers, not yet connected."""
    return McpManager(
        load_servers(settings.mcp_config_path),
        connect_timeout=settings.mcp_connect_timeout,
        call_timeout=settings.mcp_call_timeout,
    )


async def attach_mcp_tools(registry: ToolRegistry, manager: McpManager) -> int:
    """Connect the MCP servers and register whatever they expose.

    Returns the number of tools added. A server that fails to connect simply
    contributes nothing; the assistant carries on with the rest.
    """
    if not len(manager):
        return 0

    await manager.start()

    added = 0
    for server in manager.connected:
        for spec in build_specs(server):
            try:
                registry.register(spec)
                added += 1
            except DuplicateToolError:
                # Two servers configured under the same name, or a name that
                # collides with a built-in. Skip rather than shadow.
                logger.error(f"Skipping duplicate MCP tool {spec.name}")

    logger.info(f"Added {added} MCP tool(s) from {len(manager.connected)} server(s)")
    return added
