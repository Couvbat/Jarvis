"""Tests for tool descriptions and schema conversion (tools/schema.py)."""


from jarvis.tools.schema import (
    NAMESPACE_SEPARATOR,
    Risk,
    ToolResult,
    ToolSpec,
    from_mcp_tool,
    namespaced,
    split_name,
    to_ollama_schema,
)


def noop(**kwargs):
    return ToolResult("ok")


def spec(**overrides):
    defaults = dict(
        name="fs__read",
        description="Read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=noop,
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


class TestNamespacing:
    def test_namespaced_joins_with_the_separator(self):
        assert namespaced("fs", "read") == f"fs{NAMESPACE_SEPARATOR}read"

    def test_split_round_trips(self):
        assert split_name(namespaced("git", "commit")) == ("git", "commit")

    def test_a_bare_name_has_no_namespace(self):
        assert split_name("read") == (None, "read")

    def test_only_the_first_separator_splits(self):
        assert split_name("fs__read__raw") == ("fs", "read__raw")


class TestRisk:
    def test_risks_are_ordered(self):
        assert Risk.READ_ONLY < Risk.WRITE < Risk.DESTRUCTIVE

    def test_risks_combine_with_max(self):
        assert max(Risk.READ_ONLY, Risk.DESTRUCTIVE) is Risk.DESTRUCTIVE

    def test_names_are_stable(self):
        assert [risk.name for risk in Risk] == ["READ_ONLY", "WRITE", "DESTRUCTIVE"]


class TestToolResult:
    def test_defaults_are_a_successful_trusted_result(self):
        result = ToolResult("done")
        assert result.ok is True
        assert result.untrusted is False

    def test_error_helper(self):
        result = ToolResult.error("nope")
        assert result.ok is False
        assert result.content == "Error: nope"


class TestToolSpec:
    def test_risk_defaults_to_the_worst_case(self):
        """A capability that has not said what it does is not assumed safe."""
        assert spec().risk is Risk.DESTRUCTIVE

    def test_static_risk_is_returned_as_is(self):
        assert spec(risk=Risk.READ_ONLY).assess({}) is Risk.READ_ONLY

    def test_risk_for_can_raise_the_baseline(self):
        instance = spec(risk=Risk.WRITE, risk_for=lambda args: Risk.DESTRUCTIVE)
        assert instance.assess({}) is Risk.DESTRUCTIVE

    def test_risk_for_cannot_lower_the_baseline(self):
        """Argument inspection may only make a tool look more dangerous."""
        instance = spec(risk=Risk.WRITE, risk_for=lambda args: Risk.READ_ONLY)
        assert instance.assess({}) is Risk.WRITE

    def test_a_failing_risk_assessment_falls_back_to_the_worst_case(self):
        def explode(args):
            raise RuntimeError("cannot tell")

        assert spec(risk=Risk.READ_ONLY, risk_for=explode).assess({}) is Risk.DESTRUCTIVE

    def test_hints_default_to_empty_and_are_per_instance(self):
        first, second = spec(), spec()
        first.hints["read_only_hint"] = True
        assert second.hints == {}


class TestOllamaSchema:
    def test_shape(self):
        schema = to_ollama_schema(spec())
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "fs__read"
        assert schema["function"]["description"] == "Read a file"

    def test_input_schema_becomes_parameters(self):
        instance = spec()
        assert to_ollama_schema(instance)["function"]["parameters"] is instance.input_schema


class FakeAnnotations:
    def __init__(self, **hints):
        for key, value in hints.items():
            setattr(self, key, value)


class FakeMcpTool:
    """Shaped like ``mcp.types.Tool`` without needing the SDK installed."""

    def __init__(self, name, description="", input_schema=None, annotations=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.annotations = annotations


class TestMcpConversion:
    def test_name_is_namespaced_by_server(self):
        converted = from_mcp_tool(FakeMcpTool("read_file"), "filesystem", noop)
        assert converted.name == "filesystem__read_file"

    def test_description_and_schema_carry_over(self):
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        converted = from_mcp_tool(
            FakeMcpTool("read_file", "Reads a file", schema), "filesystem", noop
        )
        assert converted.description == "Reads a file"
        assert converted.input_schema == schema

    def test_camel_case_schema_field_is_accepted(self):
        """Wire-format MCP payloads spell it ``inputSchema``."""
        tool = FakeMcpTool("read_file")
        tool.input_schema = None
        tool.inputSchema = {"type": "object", "properties": {"p": {"type": "string"}}}
        assert from_mcp_tool(tool, "fs", noop).input_schema == tool.inputSchema

    def test_a_missing_schema_becomes_an_empty_object_schema(self):
        converted = from_mcp_tool(FakeMcpTool("ping"), "server", noop)
        assert converted.input_schema == {"type": "object", "properties": {}}

    def test_annotations_are_recorded_as_hints(self):
        tool = FakeMcpTool("read_file", annotations=FakeAnnotations(
            read_only_hint=True, destructive_hint=False, open_world_hint=True,
        ))
        converted = from_mcp_tool(tool, "filesystem", noop)
        assert converted.hints == {
            "read_only_hint": True,
            "destructive_hint": False,
            "open_world_hint": True,
        }

    def test_a_read_only_annotation_does_not_lower_the_risk(self):
        """Annotations are the server's claim about itself.

        They are recorded for the policy engine to weigh against how much the
        user trusts that server; on their own they buy nothing.
        """
        tool = FakeMcpTool("read_file", annotations=FakeAnnotations(read_only_hint=True))
        assert from_mcp_tool(tool, "filesystem", noop).risk is Risk.DESTRUCTIVE

    def test_mcp_tools_are_treated_as_egress(self):
        assert from_mcp_tool(FakeMcpTool("ping"), "server", noop).egress is True

    def test_unannotated_tools_have_no_hints(self):
        assert from_mcp_tool(FakeMcpTool("ping"), "server", noop).hints == {}
