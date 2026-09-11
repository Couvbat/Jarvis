"""Tests for conversation state and Ollama integration (llm_module.py)."""

import pytest

from llm_module import ConversationHistory, LLMModule
from tests._stubs import make_chat_response, make_tool_call


class TestConversationHistory:
    def test_starts_empty(self):
        assert ConversationHistory().get_messages() == []

    def test_add_message_shape(self):
        history = ConversationHistory()
        history.add_message("user", "hello")
        assert history.get_messages() == [{"role": "user", "content": "hello"}]

    def test_messages_keep_their_order(self):
        history = ConversationHistory(max_history=10)
        for index in range(5):
            history.add_message("user", str(index))
        assert [m["content"] for m in history.get_messages()] == list("01234")

    def test_truncation_keeps_the_system_message(self):
        history = ConversationHistory(max_history=4)
        history.add_message("system", "SYSTEM")
        for index in range(20):
            history.add_message("user", str(index))
        messages = history.get_messages()
        assert messages[0] == {"role": "system", "content": "SYSTEM"}

    def test_truncation_caps_the_length(self):
        history = ConversationHistory(max_history=4)
        history.add_message("system", "SYSTEM")
        for index in range(20):
            history.add_message("user", str(index))
        assert len(history.get_messages()) == 5

    def test_truncation_keeps_the_most_recent_turns(self):
        history = ConversationHistory(max_history=4)
        history.add_message("system", "SYSTEM")
        for index in range(20):
            history.add_message("user", str(index))
        assert [m["content"] for m in history.get_messages()[1:]] == [
            "16", "17", "18", "19",
        ]

    def test_no_truncation_below_the_limit(self):
        history = ConversationHistory(max_history=10)
        for index in range(8):
            history.add_message("user", str(index))
        assert len(history.get_messages()) == 8

    def test_clear_keeps_only_the_system_message(self):
        history = ConversationHistory()
        history.add_message("system", "SYSTEM")
        history.add_message("user", "hello")
        history.clear()
        assert history.get_messages() == [{"role": "system", "content": "SYSTEM"}]

    def test_clear_on_empty_history_is_safe(self):
        history = ConversationHistory()
        history.clear()
        assert history.get_messages() == []

    def test_get_messages_returns_the_live_internal_list(self):
        """Callers can mutate the history through the accessor.

        Harmless today because every caller only reads it, but it means the
        history has no ownership boundary.
        """
        history = ConversationHistory()
        history.add_message("user", "hello")
        history.get_messages().append({"role": "user", "content": "injected"})
        assert len(history.get_messages()) == 2

    def test_clear_without_a_system_message_keeps_the_first_turn(self):
        """``clear()`` assumes index 0 is the system prompt; it is not always."""
        history = ConversationHistory()
        history.add_message("user", "first")
        history.add_message("user", "second")
        history.clear()
        assert history.get_messages() == [{"role": "user", "content": "first"}]

    @pytest.mark.xfail(strict=True, reason="BUG-07: max_history=0 disables truncation")
    def test_zero_max_history_should_truncate(self):
        """``messages[-0:]`` is ``messages[0:]`` - the whole list.

        Setting MAX_CONVERSATION_HISTORY=0 therefore grows the context without
        bound instead of keeping only the system prompt.
        """
        history = ConversationHistory(max_history=0)
        history.add_message("system", "SYSTEM")
        for index in range(20):
            history.add_message("user", str(index))
        assert len(history.get_messages()) <= 2


class TestLLMModuleSetup:
    def test_seeds_the_system_prompt(self):
        module = LLMModule()
        messages = module.history.get_messages()
        assert messages[0]["role"] == "system"
        assert "Jarvis" in messages[0]["content"]

    def test_reads_settings(self, settings):
        settings.ollama_model = "mistral:7b"
        settings.llm_temperature = 0.1
        module = LLMModule()
        assert module.model == "mistral:7b"
        assert module.temperature == pytest.approx(0.1)

    def test_tool_schemas_are_well_formed(self):
        for tool in LLMModule.TOOLS:
            assert tool["type"] == "function"
            function = tool["function"]
            assert function["name"] and function["description"]
            parameters = function["parameters"]
            assert parameters["type"] == "object"
            for required in parameters["required"]:
                assert required in parameters["properties"]

    def test_tool_names_are_unique(self):
        names = [tool["function"]["name"] for tool in LLMModule.TOOLS]
        assert len(names) == len(set(names))


class TestChat:
    def test_returns_plain_content(self, fake_ollama):
        fake_ollama.responses.append(make_chat_response("Bonjour"))
        result = LLMModule().chat("salut")
        assert result["response"] == "Bonjour"
        assert result["tool_calls"] is None

    def test_sends_model_tools_and_options(self, fake_ollama, settings):
        settings.ollama_model = "llama3.1:8b"
        fake_ollama.responses.append(make_chat_response("ok"))
        LLMModule().chat("hello")
        call = fake_ollama.calls[0]
        assert call["model"] == "llama3.1:8b"
        assert call["tools"] == LLMModule.TOOLS
        assert call["options"]["temperature"] == pytest.approx(settings.llm_temperature)
        assert call["options"]["num_predict"] == settings.llm_max_tokens

    def test_sends_the_full_history(self, fake_ollama):
        fake_ollama.responses.extend([
            make_chat_response("one"), make_chat_response("two"),
        ])
        module = LLMModule()
        module.chat("first")
        module.chat("second")
        roles = [m["role"] for m in fake_ollama.calls[1]["messages"]]
        assert roles == ["system", "user", "assistant", "user"]

    def test_records_both_turns_in_history(self, fake_ollama):
        fake_ollama.responses.append(make_chat_response("Bonjour"))
        module = LLMModule()
        module.chat("salut")
        messages = module.history.get_messages()
        assert messages[-2] == {"role": "user", "content": "salut"}
        assert messages[-1] == {"role": "assistant", "content": "Bonjour"}

    def test_returns_tool_calls(self, fake_ollama):
        call = make_tool_call("fetch_web_page", {"url": "https://example.com"})
        fake_ollama.responses.append(make_chat_response("on it", [call]))
        result = LLMModule().chat("read example.com")
        assert result["tool_calls"] == [call]

    def test_transport_error_is_converted_to_a_message(self, fake_ollama):
        fake_ollama.error = ConnectionError("ollama is down")
        result = LLMModule().chat("hello")
        assert "error" in result["response"].lower()
        assert result["tool_calls"] is None

    def test_transport_error_still_records_a_turn(self, fake_ollama):
        fake_ollama.error = ConnectionError("ollama is down")
        module = LLMModule()
        module.chat("hello")
        assert module.history.get_messages()[-1]["role"] == "assistant"

    def test_missing_message_key_does_not_raise(self, fake_ollama):
        fake_ollama.responses.append({})
        result = LLMModule().chat("hello")
        assert result["response"] == ""

    def test_empty_tool_call_list_is_normalised_to_none(self, fake_ollama):
        fake_ollama.responses.append(make_chat_response("hi", []))
        assert LLMModule().chat("hello")["tool_calls"] is None


class TestToolResults:
    def test_tool_result_is_appended(self, fake_ollama):
        module = LLMModule()
        module.add_tool_result("fetch_web_page", "some content")
        last = module.history.get_messages()[-1]
        assert last["role"] == "tool"
        assert "fetch_web_page" in last["content"]
        assert "some content" in last["content"]

    def test_reset_drops_the_conversation(self, fake_ollama):
        fake_ollama.responses.append(make_chat_response("hi"))
        module = LLMModule()
        module.chat("hello")
        module.reset_conversation()
        assert all(m["role"] == "system" for m in module.history.get_messages())


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True, reason="BUG-08: tool calls from a modern ollama client break chat()"
)
def test_tool_call_without_content_survives_history_recording(fake_ollama):
    """``json.dumps(tool_calls)`` cannot serialise the client's pydantic models.

    When the model answers with tool calls and no prose - the normal case -
    ``chat()`` raises inside its own try block and degrades to the generic
    error string, so the tools are never executed.
    """
    call = make_tool_call("fetch_web_page", {"url": "https://example.com"})
    fake_ollama.responses.append(make_chat_response("", [call]))
    result = LLMModule().chat("read example.com")
    assert result["tool_calls"] == [call]


@pytest.mark.xfail(
    strict=True, reason="BUG-09: assistant tool calls are stored as prose, not structure"
)
def test_tool_calls_are_replayed_in_the_protocol_shape(fake_ollama):
    """Ollama expects ``{"role": "assistant", "tool_calls": [...]}`` followed by
    ``{"role": "tool", "content": ...}``.  Jarvis flattens both into strings, so
    the model cannot correlate a result with the call that produced it."""
    call = make_tool_call("fetch_web_page", {"url": "https://example.com"}, as_model=False)
    fake_ollama.responses.extend([
        make_chat_response("", [call], as_model=False), make_chat_response("done"),
    ])
    module = LLMModule()
    module.chat("read example.com")
    module.add_tool_result("fetch_web_page", "content")
    module.chat("summarise")
    replayed = fake_ollama.calls[-1]["messages"]
    assert any(m.get("tool_calls") for m in replayed)


@pytest.mark.xfail(
    strict=True, reason="BUG-11: reset_conversation duplicates the system prompt"
)
def test_reset_conversation_keeps_exactly_one_system_prompt(fake_ollama):
    """``clear()`` already keeps message 0 (the system prompt) and
    ``reset_conversation`` then appends another one.  Each reset adds a copy."""
    fake_ollama.responses.append(make_chat_response("hi"))
    module = LLMModule()
    module.chat("hello")
    module.reset_conversation()
    module.reset_conversation()
    assert len(module.history.get_messages()) == 1


@pytest.mark.xfail(strict=True, reason="BUG-10: no health check against the Ollama server")
def test_module_exposes_a_health_check(fake_ollama):
    """Without one, a stopped Ollama only surfaces as a spoken error per turn."""
    assert LLMModule().is_available() in (True, False)
