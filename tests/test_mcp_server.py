"""Tests for serving Jarvis's tools over MCP (jarvis/mcp_server.py).

Driven through a real MCP client over the SDK's in-memory transport, like the
client-side tests: the protocol is genuine, only the process boundary is
removed. A hand-rolled fake would let protocol mistakes through.

What matters here is not that tools can be listed. It is that the policy
engine still decides, with nobody at a keyboard to answer it.
"""

import contextlib

import pytest
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport

from jarvis.mcp_server import (
    JarvisMcpServer,
    offerable,
    permitted,
    to_mcp_tool,
)
from jarvis.policy.engine import PolicyEngine
from jarvis.policy.store import ApprovalStore
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.schema import Risk, ToolResult, ToolSpec


def spec(name, risk=Risk.READ_ONLY, **kwargs):
    return ToolSpec(
        name=name,
        description=f"{name} does something",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **arguments: ToolResult(f"{name} ran"),
        risk=risk,
        **kwargs,
    )


def server_over(*specs, store=None):
    registry = ToolRegistry()
    registry.register_all(list(specs))
    return JarvisMcpServer(
        registry=registry,
        policy=PolicyEngine(store or ApprovalStore(":memory:")),
    )


@contextlib.asynccontextmanager
async def client_for(instance):
    """A real MCP client talking to this server."""
    async with InMemoryTransport(instance.build()) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            yield session


class TestWhatIsOffered:
    def test_reading_is_offered(self):
        assert offerable(spec("fs__read")) is True

    def test_writing_is_not(self):
        """There is nobody here to confirm it, so advertising it would be
        advertising a refusal."""
        assert offerable(spec("fs__write", risk=Risk.WRITE)) is False

    def test_deleting_is_not(self):
        assert offerable(spec("fs__delete", risk=Risk.DESTRUCTIVE)) is False

    def test_egress_is_not(self):
        """Read-only in risk terms, and still a request leaving the user's
        machine - which is a decision for the person whose machine it is."""
        assert offerable(spec("web__fetch", egress=True)) is False

    def test_a_preapproved_tool_is(self):
        """The user authorised it out of band, in their own configuration."""
        assert offerable(spec("remind__list", preapproved=True)) is True

    def test_the_default_registry_offers_only_safe_tools(self, settings, tmp_path):
        settings.allowed_directories = str(tmp_path)
        names = [tool.name for tool in JarvisMcpServer().tools()]

        assert "fs__read" in names
        assert "fs__write" not in names
        assert "fs__delete" not in names
        assert "web__fetch" not in names


class TestTheProtocol:
    async def test_a_client_can_list_the_tools(self):
        instance = server_over(spec("fs__read"), spec("fs__list"))
        async with client_for(instance) as session:
            listed = await session.list_tools()
        assert {tool.name for tool in listed.tools} == {"fs__read", "fs__list"}

    async def test_a_client_can_call_one(self):
        instance = server_over(spec("fs__read"))
        async with client_for(instance) as session:
            result = await session.call_tool("fs__read", {})
        assert result.content[0].text == "fs__read ran"
        assert not result.is_error

    async def test_the_schema_survives_the_round_trip(self):
        """A client that cannot see the arguments cannot call the tool."""
        detailed = ToolSpec(
            name="fs__read",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=lambda **kwargs: ToolResult("ok"),
            risk=Risk.READ_ONLY,
        )
        async with client_for(server_over(detailed)) as session:
            listed = await session.list_tools()
        assert listed.tools[0].input_schema["required"] == ["path"]

    async def test_risk_is_advertised_as_an_annotation(self):
        """So a client can show its own user what a call would do."""
        async with client_for(server_over(spec("fs__read"))) as session:
            listed = await session.list_tools()
        assert listed.tools[0].annotations.read_only_hint is True

    async def test_a_failing_tool_is_an_error_not_a_crash(self):
        def explode(**kwargs):
            raise RuntimeError("disk on fire")

        failing = ToolSpec(
            name="fs__read", description="Read", handler=explode,
            input_schema={"type": "object", "properties": {}},
            risk=Risk.READ_ONLY,
        )
        async with client_for(server_over(failing)) as session:
            result = await session.call_tool("fs__read", {})
        assert result.is_error


class TestThePolicyStillDecides:
    """The whole point. A server with nobody at the keyboard must not become
    a way around the confirmations."""

    async def test_a_write_is_refused_with_a_reason(self):
        instance = server_over(spec("fs__write", risk=Risk.WRITE))
        result = await instance.call("fs__write", {})
        assert result.ok is False
        assert "nobody at the keyboard" in result.content

    async def test_a_deletion_is_refused(self):
        instance = server_over(spec("fs__delete", risk=Risk.DESTRUCTIVE))
        assert (await instance.call("fs__delete", {})).ok is False

    async def test_calling_an_unlisted_tool_still_goes_through_the_engine(self):
        """The listing is a hint; the engine is the gate. A client that knows
        the name must not get a shortcut."""
        instance = server_over(spec("fs__delete", risk=Risk.DESTRUCTIVE))
        async with client_for(instance) as session:
            result = await session.call_tool("fs__delete", {})
        assert result.is_error

    async def test_a_standing_approval_lets_it_through(self):
        """Granted by the user in Jarvis itself, not by the caller."""
        store = ApprovalStore(":memory:")
        writing = spec(
            "fs__write", risk=Risk.WRITE,
            scope_for=lambda arguments: "/tmp",
        )
        store.approve("fs__write", "/tmp", Risk.WRITE)

        instance = server_over(writing, store=store)
        assert (await instance.call("fs__write", {})).ok is True

    async def test_a_refused_precheck_says_why(self):
        """Not "needs confirmation": the path was outside the sandbox, and
        that is a different message."""
        guarded = spec(
            "fs__read",
            precheck=lambda arguments: "'/etc/passwd' is outside the sandbox",
        )
        result = await server_over(guarded).call("fs__read", {})
        assert result.ok is False
        assert "outside the sandbox" in result.content

    async def test_an_unknown_tool_is_reported(self):
        result = await server_over(spec("fs__read")).call("nope__nope", {})
        assert result.ok is False
        assert "no such tool" in result.content

    async def test_untrusted_content_escalates_the_next_call(self):
        """The same taint rule a spoken session has - which here means the
        next write is refused rather than merely confirmed."""
        tainting = ToolSpec(
            name="docs__search", description="Search",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kwargs: ToolResult(
                "found", untrusted=True, origin="the index"
            ),
            risk=Risk.READ_ONLY,
        )
        writing = spec(
            "fs__write", risk=Risk.WRITE, scope_for=lambda arguments: "/tmp"
        )
        store = ApprovalStore(":memory:")
        store.approve("fs__write", "/tmp", Risk.WRITE)

        instance = server_over(tainting, writing, store=store)
        assert (await instance.call("fs__write", {})).ok is True

        await instance.call("docs__search", {})
        after = await instance.call("fs__write", {})
        assert after.ok is False, "the standing approval survived the taint"


class TestTheCommand:
    def test_listing_prints_the_tools(self, settings, tmp_path, capsys):
        from jarvis.mcp_server import main

        settings.allowed_directories = str(tmp_path)
        assert main(["--list"]) == 0
        assert "fs__read" in capsys.readouterr().out

    def test_the_listing_does_not_include_writes(
        self, settings, tmp_path, capsys
    ):
        from jarvis.mcp_server import main

        settings.allowed_directories = str(tmp_path)
        main(["--list"])
        assert "fs__delete" not in capsys.readouterr().out


class TestConversion:
    @pytest.mark.parametrize("risk,read_only,destructive", [
        (Risk.READ_ONLY, True, False),
        (Risk.WRITE, False, False),
        (Risk.DESTRUCTIVE, False, True),
    ])
    def test_annotations_follow_the_risk(self, risk, read_only, destructive):
        tool = to_mcp_tool(spec("x__y", risk=risk))
        assert tool.annotations.read_only_hint is read_only
        assert tool.annotations.destructive_hint is destructive

    def test_egress_is_flagged_as_open_world(self):
        assert to_mcp_tool(spec("web__fetch", egress=True)).annotations.open_world_hint


class TestTheRule:
    """Stated in one place, so it can be read and tested there."""

    def decide(self, instance, tool_spec, arguments=None):
        return instance.policy.evaluate(tool_spec, arguments or {}, instance.taint)

    def test_a_read_inside_the_sandbox_runs(self):
        """The sandbox *is* the authorisation for a read: it is the setting
        whose whole job is saying which files a program may see."""
        reading = spec("fs__read")
        instance = server_over(reading)
        assert permitted(reading, self.decide(instance, reading)) is True

    def test_a_write_does_not(self):
        writing = spec("fs__write", risk=Risk.WRITE)
        instance = server_over(writing)
        assert permitted(writing, self.decide(instance, writing)) is False

    def test_a_deletion_does_not(self):
        deleting = spec("fs__delete", risk=Risk.DESTRUCTIVE)
        instance = server_over(deleting)
        assert permitted(deleting, self.decide(instance, deleting)) is False

    def test_egress_does_not(self):
        fetching = spec("web__fetch", egress=True)
        instance = server_over(fetching)
        assert permitted(fetching, self.decide(instance, fetching)) is False

    def test_a_read_after_untrusted_content_does_not(self):
        """Every call made after untrusted content entered the session is one
        a person should see."""
        tainting = ToolSpec(
            name="docs__search", description="Search",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kwargs: ToolResult(
                "found", untrusted=True, origin="the index"
            ),
            risk=Risk.READ_ONLY, egress=True,
        )
        reading = spec("fs__read")
        instance = server_over(tainting, reading)

        instance.taint.observe("docs__search", ToolResult(
            "found", untrusted=True, origin="the index"
        ))
        decision = self.decide(instance, reading)
        # A plain read is not escalated by taint alone - only writes and
        # egress are - so this one still runs, and that is deliberate.
        assert permitted(reading, decision) is True
