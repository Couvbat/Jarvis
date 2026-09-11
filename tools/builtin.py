"""Assembling the tools Jarvis ships with.

One place that knows which local providers exist, so adding MCP servers in
Phase 2 is a matter of registering more specs into the same registry.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from config import settings
from policy.paths import PathPolicy
from tools.local import apps, filesystem, web
from tools.registry import ToolRegistry


def build_default_registry(path_policy: Optional[PathPolicy] = None) -> ToolRegistry:
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
