"""Tests for MCP connection handling (tools/mcp/manager.py).

These drive a real MCP server over the SDK's in-memory transport: the protocol
is genuine, only the process boundary is removed. A hand-rolled fake would let
protocol mistakes through.
"""

import asyncio
import contextlib

from mcp import ClientSession
from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer

from tools.mcp.manager import McpManager, McpServer, ServerStatus, result_text
from tools.mcp.servers import ServerConfig, Transport


def build_server(name="probe"):
    """A small but real MCP server."""
    server = MCPServer(name=name, version="1.0.0")

    @server.tool()
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    @server.tool()
    def echo(text: str) -> str:
        """Echo the text back."""
        return text

    @server.tool()
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("server side failure")

    return server


@contextlib.asynccontextmanager
async def connected(config=None, server=None):
    """An McpServer attached to an in-process MCP server."""
    config = config or ServerConfig("probe", Transport.STDIO, command="unused")
    async with InMemoryTransport(server or build_server(config.name)) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            init = await session.initialize()
            instance = McpServer(config)
            await instance.attach_session(init.server_info, session)
            yield instance


class TestConnectionAndDiscovery:
    async def test_tools_are_discovered(self):
        async with connected() as server:
            assert set(server.tools) == {"probe__add", "probe__echo", "probe__boom"}

    async def test_tool_names_are_namespaced_by_server(self):
        config = ServerConfig("filesystem", Transport.STDIO, command="unused")
        async with connected(config, build_server("filesystem")) as server:
            assert all(name.startswith("filesystem__") for name in server.tools)

    async def test_two_servers_can_expose_the_same_tool_name(self):
        first = ServerConfig("alpha", Transport.STDIO, command="unused")
        second = ServerConfig("beta", Transport.STDIO, command="unused")
        async with connected(first, build_server("alpha")) as a:
            async with connected(second, build_server("beta")) as b:
                assert "alpha__add" in a.tools
                assert "beta__add" in b.tools

    async def test_the_schema_survives_the_round_trip(self):
        async with connected() as server:
            schema = server.tools["probe__add"].input_schema
            assert schema["type"] == "object"
            assert set(schema["required"]) == {"a", "b"}

    async def test_the_status_reflects_the_connection(self):
        async with connected() as server:
            assert server.status is ServerStatus.CONNECTED


class TestCalling:
    async def test_a_successful_call(self):
        async with connected() as server:
            result = await server.call("probe__add", {"a": 2, "b": 3})
            assert result.ok is True
            assert result.content == "5"

    async def test_results_are_untrusted(self):
        """A third-party server's output is content the user did not write."""
        async with connected() as server:
            result = await server.call("probe__echo", {"text": "hello"})
            assert result.untrusted is True

    async def test_a_server_side_error_is_a_failed_result(self):
        async with connected() as server:
            result = await server.call("probe__boom", {})
            assert result.ok is False
            assert "Error" in result.content

    async def test_the_origin_is_the_server(self):
        """So reading more from the same server is not treated as carrying
        its content somewhere new."""
        async with connected() as server:
            result = await server.call("probe__echo", {"text": "hi"})
            assert result.origin == "probe"

    async def test_an_error_result_is_still_untrusted(self):
        """Error text comes from the server too, so it taints like any other."""
        async with connected() as server:
            assert (await server.call("probe__boom", {})).untrusted is True

    async def test_an_unknown_tool_is_a_result_not_an_exception(self):
        async with connected() as server:
            result = await server.call("probe__missing", {})
            assert result.ok is False

    async def test_calling_a_disconnected_server_is_reported(self):
        server = McpServer(ServerConfig("probe", Transport.STDIO, command="unused"))
        result = await server.call("probe__add", {"a": 1, "b": 2})
        assert result.ok is False
        assert "not connected" in result.content

    async def test_a_slow_call_times_out(self, monkeypatch):
        config = ServerConfig("probe", Transport.STDIO, command="unused")
        async with connected(config) as server:
            server.call_timeout = 0.01

            async def never_returns(*args, **kwargs):
                await asyncio.sleep(10)

            monkeypatch.setattr(server._group, "call_tool", never_returns)
            result = await server.call("probe__add", {"a": 1, "b": 2})
            assert result.ok is False
            assert "timed out" in result.content

    async def test_a_transport_failure_mid_call_is_reported(self, monkeypatch):
        async with connected() as server:
            async def explode(*args, **kwargs):
                raise ConnectionResetError("server went away")

            monkeypatch.setattr(server._group, "call_tool", explode)
            result = await server.call("probe__add", {"a": 1, "b": 2})
            assert result.ok is False
            assert "server went away" in result.content


class TestResultFlattening:
    def test_text_blocks_are_joined(self):
        class Block:
            def __init__(self, text):
                self.text = text

        class Result:
            content = [Block("one"), Block("two")]

        assert result_text(Result()) == "one\ntwo"

    def test_non_text_blocks_are_described(self):
        class Image:
            type = "image"

        class Result:
            content = [Image()]

        assert "image content omitted" in result_text(Result())

    def test_structured_content_is_used_when_there_are_no_blocks(self):
        class Result:
            content = []
            structured_content = {"answer": 42}

        assert "42" in result_text(Result())

    def test_an_empty_result_is_an_empty_string(self):
        class Result:
            content = []

        assert result_text(Result()) == ""


class TestLifecycleFailures:
    async def test_a_command_that_does_not_exist_fails_the_server_only(self):
        config = ServerConfig(
            "ghost", Transport.STDIO, command="definitely-not-a-real-binary-xyz"
        )
        server = McpServer(config, connect_timeout=5.0)
        assert await server.start() is False
        assert server.status is ServerStatus.FAILED
        assert server.error
        await server.stop()

    async def test_a_server_that_never_connects_times_out(self, monkeypatch):
        config = ServerConfig("slow", Transport.STDIO, command="unused")
        server = McpServer(config, connect_timeout=0.05)

        async def hang(*args, **kwargs):
            await asyncio.sleep(10)

        monkeypatch.setattr(
            "tools.mcp.manager.ClientSessionGroup.connect_to_server", hang
        )
        assert await server.start() is False
        assert "did not connect" in server.error

    async def test_giving_up_on_a_stuck_server_is_immediate(self, monkeypatch):
        """It never connected, so there is no session to close gracefully.

        Waiting out the shutdown grace period here would delay every other
        server queued behind it at startup.
        """
        config = ServerConfig("slow", Transport.STDIO, command="unused")
        server = McpServer(config, connect_timeout=0.05)

        async def hang(*args, **kwargs):
            await asyncio.sleep(30)

        monkeypatch.setattr(
            "tools.mcp.manager.ClientSessionGroup.connect_to_server", hang
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        await server.start()
        await server.stop()
        assert loop.time() - started < 1.0

    async def test_a_disabled_server_is_not_started(self):
        config = ServerConfig("off", Transport.STDIO, command="unused", enabled=False)
        server = McpServer(config)
        assert await server.start() is False
        assert server.status is ServerStatus.DISABLED

    async def test_stopping_a_server_that_never_started_is_safe(self):
        await McpServer(ServerConfig("x", Transport.STDIO, command="unused")).stop()


class TestManager:
    def config(self, name, **kwargs):
        return ServerConfig(name, Transport.STDIO, command="unused", **kwargs)

    async def test_one_failing_server_does_not_stop_the_others(self, monkeypatch):
        """The whole point of per-server isolation."""
        good = self.config("good")
        bad = ServerConfig("bad", Transport.STDIO, command="definitely-not-real-xyz")

        manager = McpManager([good, bad], connect_timeout=5.0)

        async def connect(self_group, params, session_params=None):
            if params.command == "unused":
                return None  # pretend success; tools stay empty
            raise FileNotFoundError(params.command)

        monkeypatch.setattr(
            "tools.mcp.manager.ClientSessionGroup.connect_to_server", connect
        )
        await manager.start()

        statuses = manager.statuses()
        assert statuses["good"] == "connected"
        assert statuses["bad"].startswith("failed")
        await manager.stop()

    async def test_an_empty_configuration_is_fine(self):
        manager = McpManager([])
        await manager.start()
        assert manager.connected == []
        await manager.stop()

    async def test_server_for_resolves_the_namespace(self):
        manager = McpManager([self.config("git"), self.config("files")])
        assert manager.server_for("git__commit").namespace == "git"
        assert manager.server_for("files__read").namespace == "files"

    async def test_server_for_an_unknown_namespace_is_none(self):
        manager = McpManager([self.config("git")])
        assert manager.server_for("nope__thing") is None

    async def test_disabled_servers_are_reported_as_such(self):
        manager = McpManager([self.config("off", enabled=False)])
        await manager.start()
        assert manager.statuses() == {"off": "disabled"}
        await manager.stop()

    async def test_length_counts_configured_servers(self):
        assert len(McpManager([self.config("a"), self.config("b")])) == 2
