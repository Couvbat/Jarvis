"""Tests for assembling the tool set (tools/builtin.py)."""


from policy.paths import PathPolicy
from tools.builtin import attach_mcp_tools, build_default_registry, build_mcp_manager
from tools.mcp.manager import McpManager, ServerStatus
from tools.mcp.servers import ServerConfig, Transport, Trust
from tools.registry import ToolRegistry
from tools.schema import Risk, ToolResult, ToolSpec


class FakeMcpServer:
    def __init__(self, namespace, tool_names, connected=True):
        self.config = ServerConfig(namespace, Transport.STDIO, command="x",
                                   trust=Trust.CONFIRM)
        self.namespace = namespace
        self.is_connected = connected
        self.status = ServerStatus.CONNECTED if connected else ServerStatus.FAILED
        self.tools = {
            f"{namespace}__{name}": _FakeTool(name) for name in tool_names
        }

    async def call(self, qualified_name, arguments):
        return ToolResult("ok")


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} does a thing"
        self.input_schema = {"type": "object", "properties": {}}
        self.annotations = None


class FakeManager:
    def __init__(self, servers):
        self.servers = servers
        self.started = False

    def __len__(self):
        return len(self.servers)

    @property
    def connected(self):
        return [s for s in self.servers if s.is_connected]

    def statuses(self):
        return {s.namespace: s.status.value for s in self.servers}

    async def start(self):
        self.started = True

    async def stop(self):
        pass


class TestBuiltInRegistry:
    def test_the_three_local_providers_are_registered(self, settings, tmp_path):
        settings.allowed_directories = str(tmp_path)
        registry = build_default_registry()
        namespaces = {name.split("__")[0] for name in registry.names()}
        assert namespaces == {"fs", "web", "app"}

    def test_the_sandbox_can_be_supplied(self, tmp_path):
        registry = build_default_registry(PathPolicy([tmp_path]))
        assert registry.get("fs__read").precheck({"path": "/etc/passwd"})

    def test_names_are_unique(self, settings, tmp_path):
        settings.allowed_directories = str(tmp_path)
        names = build_default_registry().names()
        assert len(names) == len(set(names))


class TestMcpAttachment:
    async def test_nothing_configured_does_nothing(self):
        registry = ToolRegistry()
        assert await attach_mcp_tools(registry, FakeManager([])) == 0

    async def test_tools_from_a_connected_server_are_registered(self):
        registry = ToolRegistry()
        manager = FakeManager([FakeMcpServer("notes", ["read", "write"])])
        assert await attach_mcp_tools(registry, manager) == 2
        assert set(registry.names()) == {"notes__read", "notes__write"}

    async def test_the_manager_is_started(self):
        manager = FakeManager([FakeMcpServer("notes", ["read"])])
        await attach_mcp_tools(ToolRegistry(), manager)
        assert manager.started is True

    async def test_a_failed_server_contributes_nothing(self):
        """The assistant carries on with whatever else connected."""
        registry = ToolRegistry()
        manager = FakeManager([
            FakeMcpServer("good", ["read"]),
            FakeMcpServer("bad", ["read"], connected=False),
        ])
        assert await attach_mcp_tools(registry, manager) == 1
        assert registry.names() == ["good__read"]

    async def test_a_name_clash_is_skipped_rather_than_shadowed(self):
        """Two servers under one name, or a name that collides with a
        built-in: skip it, never let one quietly replace the other."""
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="notes__read",
            description="the built-in one",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kwargs: ToolResult("built-in"),
            risk=Risk.READ_ONLY,
        ))
        manager = FakeManager([FakeMcpServer("notes", ["read", "write"])])

        assert await attach_mcp_tools(registry, manager) == 1
        assert registry.get("notes__read").description == "the built-in one"

    async def test_local_and_mcp_tools_share_one_registry(self, settings, tmp_path):
        settings.allowed_directories = str(tmp_path)
        registry = build_default_registry()
        before = len(registry.names())
        await attach_mcp_tools(registry, FakeManager([FakeMcpServer("notes", ["read"])]))
        assert len(registry.names()) == before + 1
        assert "notes__read" in registry.names()


class TestManagerFromSettings:
    def test_a_missing_configuration_gives_an_empty_manager(self, settings, tmp_path):
        settings.mcp_config_path = str(tmp_path / "absent.json")
        assert len(build_mcp_manager()) == 0

    def test_timeouts_come_from_settings(self, settings, tmp_path):
        import json

        path = tmp_path / "mcp_servers.json"
        path.write_text(json.dumps({"mcpServers": {"x": {"command": "run-me"}}}))
        settings.mcp_config_path = str(path)
        settings.mcp_connect_timeout = 3.0
        settings.mcp_call_timeout = 7.0

        manager = build_mcp_manager()
        assert isinstance(manager, McpManager)
        assert manager.servers[0].connect_timeout == 3.0
        assert manager.servers[0].call_timeout == 7.0
