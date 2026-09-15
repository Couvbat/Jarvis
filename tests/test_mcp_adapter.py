"""Tests for MCP tools entering the registry (tools/mcp/adapter.py).

The question these pin down: how much is a server's word about its own tools
worth? Annotations may only make a tool look more dangerous; what makes one
look safer is the trust level the user set.
"""

import contextlib

import pytest
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from policy.engine import PolicyEngine, Surface
from policy.store import ApprovalStore
from policy.taint import TaintState
from tools.mcp.adapter import build_specs, risk_for
from tools.mcp.manager import McpServer, ServerStatus
from tools.mcp.servers import ServerConfig, Transport, Trust
from tools.registry import ToolRegistry
from tools.schema import Risk


class FakeAnnotations:
    def __init__(self, **hints):
        for key, value in hints.items():
            setattr(self, key, value)


class FakeTool:
    def __init__(self, name="thing", annotations=None, description="does a thing"):
        self.name = name
        self.description = description
        self.input_schema = {"type": "object", "properties": {}}
        self.annotations = annotations


def annotated(name="thing", **hints):
    return FakeTool(name, annotations=FakeAnnotations(**hints))


# --------------------------------------------------------------------------- #
# Risk mapping
# --------------------------------------------------------------------------- #

class TestConfirmTrust:
    """The default. Every call confirmed, and approvable per tool."""

    def test_an_unannotated_tool_is_a_write(self):
        assert risk_for(FakeTool(), Trust.CONFIRM) is Risk.WRITE

    def test_a_read_only_claim_buys_nothing(self):
        """Relaxation comes from trust, not from what the server says."""
        assert risk_for(annotated(read_only_hint=True), Trust.CONFIRM) is Risk.WRITE

    def test_a_destructive_claim_still_hardens(self):
        assert risk_for(annotated(destructive_hint=True), Trust.CONFIRM) \
            is Risk.DESTRUCTIVE


class TestTrustedTrust:
    def test_a_read_only_tool_is_read_only(self):
        assert risk_for(annotated(read_only_hint=True), Trust.TRUSTED) is Risk.READ_ONLY

    def test_an_unannotated_tool_is_still_a_write(self):
        assert risk_for(FakeTool(), Trust.TRUSTED) is Risk.WRITE

    def test_a_destructive_claim_overrides_trust(self):
        """Hardening applies at every trust level."""
        assert risk_for(annotated(destructive_hint=True), Trust.TRUSTED) \
            is Risk.DESTRUCTIVE

    def test_a_tool_claiming_both_is_treated_as_destructive(self):
        tool = annotated(read_only_hint=True, destructive_hint=True)
        assert risk_for(tool, Trust.TRUSTED) is Risk.DESTRUCTIVE


class TestReadonlyTrust:
    def test_read_only_tools_are_exposed(self):
        assert risk_for(annotated(read_only_hint=True), Trust.READONLY) is Risk.READ_ONLY

    def test_unannotated_tools_are_not_exposed_at_all(self):
        """Saying nothing about itself does not qualify as read-only."""
        assert risk_for(FakeTool(), Trust.READONLY) is None

    def test_a_tool_marked_not_read_only_is_not_exposed(self):
        assert risk_for(annotated(read_only_hint=False), Trust.READONLY) is None

    def test_a_destructive_tool_is_not_exposed(self):
        assert risk_for(annotated(destructive_hint=True), Trust.READONLY) is None


# --------------------------------------------------------------------------- #
# Spec building
# --------------------------------------------------------------------------- #

class StubServer:
    def __init__(self, tools, trust=Trust.CONFIRM, connected=True, namespace="srv"):
        self.config = ServerConfig(namespace, Transport.STDIO, command="x", trust=trust)
        self.tools = tools
        self.is_connected = connected
        self.status = ServerStatus.CONNECTED if connected else ServerStatus.FAILED
        self.calls = []

    @property
    def namespace(self):
        return self.config.name

    async def call(self, qualified_name, arguments):
        from tools.schema import ToolResult

        self.calls.append((qualified_name, arguments))
        return ToolResult("from the server", untrusted=True)


class TestSpecBuilding:
    def test_names_keep_the_server_namespace(self):
        specs = build_specs(StubServer({"srv__thing": FakeTool("thing")}))
        assert [spec.name for spec in specs] == ["srv__thing"]

    def test_the_spec_name_is_the_name_the_handler_calls(self):
        """The session group assigns the name; re-deriving it would let the
        spec and its handler drift apart."""
        specs = build_specs(StubServer({"srv__renamed": FakeTool("original")}))
        assert specs[0].name == "srv__renamed"

    def test_the_description_and_schema_carry_over(self):
        spec = build_specs(StubServer({"srv__thing": FakeTool("thing")}))[0]
        assert spec.description == "does a thing"
        assert spec.input_schema["type"] == "object"

    def test_every_mcp_tool_counts_as_egress(self):
        spec = build_specs(StubServer({"srv__thing": FakeTool("thing")}))[0]
        assert spec.egress is True

    def test_hints_are_recorded_on_the_spec(self):
        tool = annotated(read_only_hint=True, open_world_hint=True)
        spec = build_specs(StubServer({"srv__thing": tool}))[0]
        assert spec.hints["read_only_hint"] is True
        assert spec.hints["open_world_hint"] is True

    def test_tools_a_readonly_server_does_not_mark_are_dropped(self):
        specs = build_specs(StubServer(
            {
                "srv__peek": annotated("peek", read_only_hint=True),
                "srv__write": FakeTool("write"),
            },
            trust=Trust.READONLY,
        ))
        assert [spec.name for spec in specs] == ["srv__peek"]

    async def test_the_handler_calls_the_server(self):
        server = StubServer({"srv__thing": FakeTool("thing")})
        spec = build_specs(server)[0]
        result = await spec.handler(query="hello")
        assert server.calls == [("srv__thing", {"query": "hello"})]
        assert result.untrusted is True

    def test_the_approval_scope_is_the_tool_itself(self):
        """There is no way to know what an MCP tool's arguments mean.

        Scoping by exact arguments would almost never match twice - a search
        with a different query is a different scope - so the approval would be
        useless while the prompting continued.
        """
        spec = build_specs(StubServer({"srv__search": FakeTool("search")}))[0]
        assert spec.scope_for({"q": "a"}) == spec.scope_for({"q": "b"}) == "srv__search"

    def test_a_trusted_servers_read_only_tools_are_preapproved(self):
        """Marking a server trusted in the config is the approval; asking
        again on first use is the prompt fatigue this design avoids."""
        spec = build_specs(StubServer(
            {"srv__peek": annotated("peek", read_only_hint=True)}, trust=Trust.TRUSTED,
        ))[0]
        assert spec.preapproved is True

    def test_a_trusted_servers_other_tools_are_not_preapproved(self):
        spec = build_specs(StubServer(
            {"srv__write": FakeTool("write")}, trust=Trust.TRUSTED,
        ))[0]
        assert spec.preapproved is False

    def test_a_confirm_server_preapproves_nothing(self):
        spec = build_specs(StubServer(
            {"srv__peek": annotated("peek", read_only_hint=True)}, trust=Trust.CONFIRM,
        ))[0]
        assert spec.preapproved is False

    def test_a_readonly_server_preapproves_nothing(self):
        spec = build_specs(StubServer(
            {"srv__peek": annotated("peek", read_only_hint=True)}, trust=Trust.READONLY,
        ))[0]
        assert spec.preapproved is False

    def test_a_disconnected_server_refuses_before_prompting(self):
        spec = build_specs(StubServer({"srv__thing": FakeTool("thing")}, connected=False))[0]
        assert "not connected" in spec.precheck({})

    def test_a_connected_server_passes_the_precheck(self):
        spec = build_specs(StubServer({"srv__thing": FakeTool("thing")}))[0]
        assert spec.precheck({}) is None


# --------------------------------------------------------------------------- #
# Through the registry and the policy engine
# --------------------------------------------------------------------------- #

@pytest.fixture
def engine():
    store = ApprovalStore()
    yield PolicyEngine(store)
    store.close()


class TestThroughPolicy:
    def spec_for(self, tool, trust):
        return build_specs(StubServer({"srv__thing": tool}, trust=trust))[0]

    def test_a_confirm_server_asks_before_the_first_call(self, engine):
        spec = self.spec_for(FakeTool("thing"), Trust.CONFIRM)
        assert engine.evaluate(spec, {}).surface is Surface.VOICE

    def test_approving_a_confirm_tool_stops_the_asking(self, engine):
        spec = self.spec_for(FakeTool("thing"), Trust.CONFIRM)
        engine.remember(engine.evaluate(spec, {"q": "first"}))
        assert engine.evaluate(spec, {"q": "second"}).is_automatic is True

    def test_a_destructive_tool_always_needs_the_keyboard(self, engine):
        spec = self.spec_for(annotated(destructive_hint=True), Trust.CONFIRM)
        decision = engine.evaluate(spec, {})
        assert decision.surface is Surface.TERMINAL
        engine.remember(decision)
        assert engine.evaluate(spec, {}).surface is Surface.TERMINAL

    def test_a_trusted_read_only_tool_needs_no_prompt_at_all(self, engine):
        spec = self.spec_for(annotated(read_only_hint=True), Trust.TRUSTED)
        assert engine.evaluate(spec, {}).is_automatic is True

    def test_an_mcp_call_is_escalated_once_the_turn_is_tainted(self, engine):
        """Every MCP tool is egress, so a tainted turn escalates it."""
        from tools.schema import ToolResult

        spec = self.spec_for(annotated(read_only_hint=True), Trust.TRUSTED)
        assert engine.evaluate(spec, {}).is_automatic is True

        taint = TaintState()
        taint.observe("web__fetch", ToolResult("page", untrusted=True))
        assert engine.evaluate(spec, {}, taint).surface is Surface.TERMINAL


# --------------------------------------------------------------------------- #
# End to end against a real server
# --------------------------------------------------------------------------- #

def build_real_server():
    server = MCPServer(name="probe", version="1.0.0")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def peek(what: str) -> str:
        """Look something up."""
        return f"saw {what}"

    @server.tool(annotations=ToolAnnotations(destructive_hint=True))
    def wipe() -> str:
        """Destroy something."""
        return "gone"

    @server.tool()
    def plain() -> str:
        """Unannotated."""
        return "ok"

    return server


@contextlib.asynccontextmanager
async def connected(trust=Trust.CONFIRM):
    config = ServerConfig("probe", Transport.STDIO, command="unused", trust=trust)
    async with InMemoryTransport(build_real_server()) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            init = await session.initialize()
            server = McpServer(config)
            await server.attach_session(init.server_info, session)
            yield server


class TestAgainstARealServer:
    async def test_tools_reach_the_registry(self):
        async with connected() as server:
            registry = ToolRegistry()
            registry.register_all(build_specs(server))
            assert set(registry.names()) == {"probe__peek", "probe__wipe", "probe__plain"}

    async def test_a_call_round_trips_through_the_registry(self):
        async with connected() as server:
            registry = ToolRegistry()
            registry.register_all(build_specs(server))
            result = await registry.call("probe__peek", {"what": "the sky"})
            assert result.content == "saw the sky"
            assert result.untrusted is True

    async def test_real_annotations_drive_the_risk(self):
        async with connected(Trust.TRUSTED) as server:
            by_name = {spec.name: spec for spec in build_specs(server)}
            assert by_name["probe__peek"].risk is Risk.READ_ONLY
            assert by_name["probe__wipe"].risk is Risk.DESTRUCTIVE
            assert by_name["probe__plain"].risk is Risk.WRITE

    async def test_a_readonly_server_exposes_only_its_read_only_tool(self):
        async with connected(Trust.READONLY) as server:
            assert [spec.name for spec in build_specs(server)] == ["probe__peek"]

    async def test_a_server_side_failure_comes_back_as_a_result(self):
        async with connected() as server:
            registry = ToolRegistry()
            registry.register_all(build_specs(server))
            result = await registry.call("probe__peek", {})  # missing argument
            assert result.ok is False
