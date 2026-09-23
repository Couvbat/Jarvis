"""Tests for tool registration and dispatch (tools/registry.py)."""

import pytest

from jarvis.tools.registry import DuplicateToolError, ToolRegistry
from jarvis.tools.schema import Risk, ToolResult, ToolSpec


def make_spec(name="fs__read", handler=None, **overrides):
    defaults = dict(
        name=name,
        description=f"does {name}",
        input_schema={"type": "object", "properties": {}},
        handler=handler or (lambda **kwargs: ToolResult("ok")),
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


@pytest.fixture
def registry():
    return ToolRegistry()


class TestRegistration:
    def test_register_then_get(self, registry):
        spec = registry.register(make_spec())
        assert registry.get("fs__read") is spec

    def test_get_of_an_unknown_tool_is_none(self, registry):
        assert registry.get("nope") is None

    def test_names_preserve_registration_order(self, registry):
        registry.register_all([make_spec("fs__read"), make_spec("web__fetch")])
        assert registry.names() == ["fs__read", "web__fetch"]

    def test_duplicate_names_are_rejected(self, registry):
        """Silent shadowing would let one provider hijack another's tool."""
        registry.register(make_spec("fs__read"))
        with pytest.raises(DuplicateToolError, match="fs__read"):
            registry.register(make_spec("fs__read"))

    def test_same_bare_name_in_two_namespaces_is_fine(self, registry):
        registry.register_all([make_spec("fs__search"), make_spec("git__search")])
        assert len(registry.names()) == 2

    def test_unregister_namespace_removes_only_that_provider(self, registry):
        registry.register_all([
            make_spec("mcp_git__commit"),
            make_spec("mcp_git__status"),
            make_spec("fs__read"),
        ])
        assert registry.unregister_namespace("mcp_git") == 2
        assert registry.names() == ["fs__read"]

    def test_unregister_an_absent_namespace_is_a_noop(self, registry):
        assert registry.unregister_namespace("nothing") == 0

    def test_a_name_can_be_reused_after_its_namespace_is_dropped(self, registry):
        registry.register(make_spec("mcp_git__commit"))
        registry.unregister_namespace("mcp_git")
        registry.register(make_spec("mcp_git__commit"))  # must not raise


class TestDescribe:
    def test_describes_every_tool_by_default(self, registry):
        registry.register_all([make_spec("fs__read"), make_spec("web__fetch")])
        described = registry.describe()
        assert [item["function"]["name"] for item in described] == [
            "fs__read", "web__fetch",
        ]

    def test_describe_can_select_a_subset(self, registry):
        """Phase 2 presents only the tools relevant to a turn."""
        registry.register_all([make_spec("fs__read"), make_spec("web__fetch")])
        described = registry.describe(["web__fetch"])
        assert [item["function"]["name"] for item in described] == ["web__fetch"]

    def test_unknown_names_in_a_selection_are_skipped(self, registry):
        registry.register(make_spec("fs__read"))
        assert registry.describe(["fs__read", "ghost"])[0]["function"]["name"] == "fs__read"
        assert len(registry.describe(["fs__read", "ghost"])) == 1

    def test_an_empty_registry_describes_nothing(self, registry):
        assert registry.describe() == []


class TestCall:
    async def test_calls_the_handler_with_the_arguments(self, registry):
        seen = {}

        def handler(**kwargs):
            seen.update(kwargs)
            return ToolResult("done")

        registry.register(make_spec(handler=handler))
        assert (await registry.call("fs__read", {"path": "/tmp/a"})).content == "done"
        assert seen == {"path": "/tmp/a"}

    async def test_no_arguments_is_allowed(self, registry):
        registry.register(make_spec(handler=lambda **kwargs: ToolResult("done")))
        assert (await registry.call("fs__read")).ok is True

    async def test_an_unknown_tool_is_a_result_not_an_exception(self, registry):
        """Tool names come from a language model; a wrong one is routine."""
        result = await registry.call("ghost", {})
        assert result.ok is False
        assert "unknown tool" in result.content

    async def test_wrong_arguments_are_reported_to_the_model(self, registry):
        def handler(path):
            return ToolResult(path)

        registry.register(make_spec(handler=handler))
        result = await registry.call("fs__read", {"wrong_name": "x"})
        assert result.ok is False
        assert "invalid arguments" in result.content

    async def test_missing_required_arguments_are_reported(self, registry):
        def handler(path):
            return ToolResult(path)

        registry.register(make_spec(handler=handler))
        assert (await registry.call("fs__read", {})).ok is False

    async def test_a_handler_that_raises_becomes_an_error_result(self, registry):
        def handler(**kwargs):
            raise RuntimeError("disk on fire")

        registry.register(make_spec(handler=handler))
        result = await registry.call("fs__read", {})
        assert result.ok is False
        assert "disk on fire" in result.content

    async def test_a_handler_returning_the_wrong_type_is_caught(self, registry):
        registry.register(make_spec(handler=lambda **kwargs: "just a string"))
        result = await registry.call("fs__read", {})
        assert result.ok is False
        assert "malformed result" in result.content

    async def test_the_arguments_mapping_is_not_mutated(self, registry):
        registry.register(make_spec(handler=lambda **kwargs: ToolResult("ok")))
        arguments = {"path": "/tmp/a"}
        await registry.call("fs__read", arguments)
        assert arguments == {"path": "/tmp/a"}

    async def test_untrusted_results_survive_dispatch(self, registry):
        registry.register(make_spec(
            handler=lambda **kwargs: ToolResult("page text", untrusted=True)
        ))
        assert (await registry.call("fs__read", {})).untrusted is True


class TestAsyncHandlers:
    """MCP handlers are coroutines; local ones are blocking I/O."""

    async def test_an_async_handler_is_awaited(self, registry):
        async def handler(**kwargs):
            return ToolResult("from a coroutine")

        registry.register(make_spec(handler=handler))
        assert (await registry.call("fs__read")).content == "from a coroutine"

    async def test_an_async_handler_that_raises_is_caught(self, registry):
        async def handler(**kwargs):
            raise RuntimeError("remote server died")

        registry.register(make_spec(handler=handler))
        result = await registry.call("fs__read")
        assert result.ok is False
        assert "remote server died" in result.content

    async def test_wrong_arguments_to_an_async_handler_are_reported(self, registry):
        async def handler(path):
            return ToolResult(path)

        registry.register(make_spec(handler=handler))
        assert (await registry.call("fs__read", {"wrong": 1})).ok is False

    async def test_a_blocking_handler_does_not_stall_the_loop(self, registry):
        """Local tools block on files, subprocesses and HTTP; they must run
        off the event loop so Phase 3's barge-in can still be heard."""
        import threading

        loop_thread = threading.get_ident()
        seen = {}

        def handler(**kwargs):
            seen["thread"] = threading.get_ident()
            return ToolResult("ok")

        registry.register(make_spec(handler=handler))
        await registry.call("fs__read")
        assert seen["thread"] != loop_thread

    async def test_calls_can_run_concurrently(self, registry):
        import asyncio

        async def slow(**kwargs):
            await asyncio.sleep(0.01)
            return ToolResult("done")

        registry.register(make_spec("a__slow", handler=slow))
        registry.register(make_spec("b__slow", handler=slow))
        results = await asyncio.gather(
            registry.call("a__slow"), registry.call("b__slow")
        )
        assert [r.content for r in results] == ["done", "done"]


class TestRiskIsVisibleToCallers:
    def test_the_spec_carries_the_risk_for_the_policy_engine(self, registry):
        registry.register(make_spec(risk=Risk.READ_ONLY))
        assert registry.get("fs__read").assess({}) is Risk.READ_ONLY
