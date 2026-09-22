"""Assembling the tools Jarvis ships with.

One place that knows which local providers exist, so adding MCP servers in
Phase 2 is a matter of registering more specs into the same registry.
"""

from __future__ import annotations

from loguru import logger

from jarvis.config import settings
from jarvis.policy.paths import PathPolicy
from jarvis.tools.local import apps, documents, filesystem, web
from jarvis.tools.mcp.adapter import build_specs
from jarvis.tools.mcp.manager import McpManager
from jarvis.tools.mcp.servers import load_servers
from jarvis.tools.registry import DuplicateToolError, ToolRegistry


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
    registry.register_all(build_document_tools())

    logger.info(
        f"Registered {len(registry.names())} tools; "
        f"sandbox: {paths.describe_allowed()}"
    )
    return registry


def build_document_tools() -> list:
    """Document search, but only once there is an index to search.

    Offering a tool that can only answer "nothing is indexed" spends a slot
    in the toolbox and a paragraph of the prompt on nothing. The index is
    built out of band by `jarvis-index`, so its absence is the normal state
    for someone who has not asked for this.
    """
    from jarvis.rag.embeddings import Embedder
    from jarvis.rag.store import DocumentStore

    path = settings.documents_path
    if not path.exists():
        return []

    try:
        store = DocumentStore(path)
        _, chunks = store.counts()
    except Exception as e:
        logger.warning(f"Could not open the document index at {path}: {e}")
        return []

    if not chunks:
        return []

    logger.info(f"Document search enabled: {chunks} indexed passage(s)")
    return documents.build_tools(
        store,
        Embedder(),
        default_limit=settings.rag_top_k,
        min_similarity=settings.rag_min_similarity,
    )


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
