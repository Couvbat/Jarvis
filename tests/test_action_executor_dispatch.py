"""Tool-call dispatch in ActionExecutor.execute_tool_call."""

from tests._stubs import SubscriptableModel


def tool_call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


class TestDispatch:
    def test_routes_file_operations(self, executor, sandbox):
        result = executor.execute_tool_call(tool_call("execute_file_operation", {
            "operation": "create_file",
            "path": str(sandbox / "a.txt"),
            "content": "hello",
        }))
        assert "File created" in result
        assert (sandbox / "a.txt").read_text() == "hello"

    def test_routes_web_fetch(self, executor, monkeypatch):
        captured = {}

        def fake_fetch(url):
            captured["url"] = url
            return "fetched"

        monkeypatch.setattr(executor, "fetch_web_page", fake_fetch)
        assert executor.execute_tool_call(
            tool_call("fetch_web_page", {"url": "https://example.com"})
        ) == "fetched"
        assert captured["url"] == "https://example.com"

    def test_routes_application_launch(self, executor, monkeypatch):
        captured = {}

        def fake_launch(application, args=None):
            captured.update(application=application, args=args)
            return "launched"

        monkeypatch.setattr(executor, "launch_application", fake_launch)
        executor.execute_tool_call(
            tool_call("launch_application", {"application": "ls", "args": ["-la"]})
        )
        assert captured == {"application": "ls", "args": ["-la"]}

    def test_unknown_tool_reports_error(self, executor):
        assert "Unknown tool" in executor.execute_tool_call(tool_call("nope", {}))

    def test_accepts_ollama_style_response_objects(self, executor, sandbox):
        """The modern ollama client returns mapping-like pydantic models."""
        call = SubscriptableModel(function=SubscriptableModel(
            name="execute_file_operation",
            arguments=SubscriptableModel(
                operation="create_file", path=str(sandbox / "a.txt"), content="x"
            ),
        ))
        assert "File created" in executor.execute_tool_call(call)


class TestMalformedToolCalls:
    def test_missing_function_key_reports_error(self, executor):
        assert "Unknown tool" in executor.execute_tool_call({})

    def test_missing_arguments_do_not_raise(self, executor):
        result = executor.execute_tool_call({"function": {"name": "fetch_web_page"}})
        assert isinstance(result, str)
        assert "Error" in result

    def test_file_operation_without_path_does_not_raise(self, executor):
        result = executor.execute_tool_call(
            tool_call("execute_file_operation", {"operation": "read_file"})
        )
        assert isinstance(result, str)
        assert "Error" in result

    def test_exception_inside_a_tool_is_converted_to_a_message(
        self, executor, monkeypatch
    ):
        def explode(*args, **kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(executor, "fetch_web_page", explode)
        result = executor.execute_tool_call(
            tool_call("fetch_web_page", {"url": "https://example.com"})
        )
        assert "Error executing tool" in result
        assert "kaboom" in result

    def test_dispatch_never_raises(self, executor):
        """Every documented tool name must return a string, never propagate."""
        for name in (
            "execute_file_operation", "fetch_web_page", "launch_application", "bogus",
        ):
            assert isinstance(executor.execute_tool_call(tool_call(name, {})), str)


class TestToolSchemaAgreement:
    """The declared tool schema and the executor must not drift apart."""

    def test_every_declared_tool_is_dispatched(self, executor):
        from llm_module import LLMModule

        for tool in LLMModule.TOOLS:
            name = tool["function"]["name"]
            result = executor.execute_tool_call(tool_call(name, {}))
            assert "Unknown tool" not in result, f"{name} is declared but not dispatched"

    def test_declared_file_operations_are_all_implemented(self, executor, sandbox):
        from llm_module import LLMModule

        schema = next(
            tool for tool in LLMModule.TOOLS
            if tool["function"]["name"] == "execute_file_operation"
        )
        operations = schema["function"]["parameters"]["properties"]["operation"]["enum"]
        for operation in operations:
            result = executor.execute_file_operation(operation, str(sandbox / "probe"))
            assert "Unknown operation" not in result, f"{operation} declared but missing"
