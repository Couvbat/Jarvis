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

    def test_get_messages_returns_a_snapshot(self):
        """The accessor must not hand out a handle on the internal list."""
        history = ConversationHistory()
        history.add_message("user", "hello")
        history.get_messages().append({"role": "user", "content": "injected"})
        assert len(history.get_messages()) == 1

    def test_clear_drops_every_turn_when_there_is_no_system_prompt(self):
        history = ConversationHistory()
        history.add_message("user", "first")
        history.add_message("user", "second")
        history.clear()
        assert history.get_messages() == []

    def test_system_prompt_is_never_counted_against_the_limit(self):
        history = ConversationHistory(max_history=3)
        history.add_message("system", "SYSTEM")
        for index in range(10):
            history.add_message("user", str(index))
        messages = history.get_messages()
        assert messages[0]["role"] == "system"
        assert len(messages) == 4  # system + 3 turns

    def test_setting_the_system_prompt_twice_replaces_it(self):
        history = ConversationHistory()
        history.add_message("system", "FIRST")
        history.add_message("system", "SECOND")
        messages = history.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "SECOND"

    def test_trimming_does_not_orphan_a_tool_result(self):
        """A tool result whose assistant turn was trimmed away is dropped too.

        Left behind, it would be a result the model cannot attach to any call.
        """
        history = ConversationHistory(max_history=2)
        history.add_assistant("", [{"function": {"name": "fetch", "arguments": {}}}])
        history.add_tool_result("fetch", "the content")
        history.add_user("and then?")
        assert [message["role"] for message in history.get_messages()] == ["user"]

    def test_assistant_tool_calls_are_stored_structurally(self):
        history = ConversationHistory()
        history.add_assistant("", [{"function": {"name": "fetch", "arguments": {}}}])
        assert history.get_messages()[-1] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "fetch", "arguments": {}}}],
        }

    def test_an_assistant_turn_without_tool_calls_omits_the_key(self):
        history = ConversationHistory()
        history.add_assistant("just prose")
        assert "tool_calls" not in history.get_messages()[-1]

    def test_zero_max_history_keeps_only_the_system_prompt(self):
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

    def test_returns_tool_calls_as_plain_data(self, fake_ollama):
        call = make_tool_call("fetch_web_page", {"url": "https://example.com"})
        fake_ollama.responses.append(make_chat_response("on it", [call]))
        result = LLMModule().chat("read example.com")
        assert result["tool_calls"] == [{
            "function": {
                "name": "fetch_web_page",
                "arguments": {"url": "https://example.com"},
            }
        }]

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


class TestContinueAfterTools:
    def test_appends_no_message_of_its_own(self, fake_ollama):
        """The tool results are the new information; a synthetic user turn
        would pollute the conversation and pin the reply to its language."""
        fake_ollama.responses.extend([
            make_chat_response("hi"), make_chat_response("done"),
        ])
        module = LLMModule()
        module.chat("bonjour")
        module.add_tool_result("fetch_web_page", "content")

        before = len(module.history.get_messages())
        module.continue_after_tools()
        after = module.history.get_messages()

        assert len(after) == before + 1          # only the assistant reply
        assert after[-1]["role"] == "assistant"
        assert all(message["role"] != "user" for message in after[before:])

    def test_sends_the_tool_result_to_the_model(self, fake_ollama):
        fake_ollama.responses.extend([
            make_chat_response("hi"), make_chat_response("done"),
        ])
        module = LLMModule()
        module.chat("bonjour")
        module.add_tool_result("fetch_web_page", "the page said hello")
        module.continue_after_tools()

        sent = fake_ollama.calls[-1]["messages"]
        assert any(message["role"] == "tool"
                   and message["content"] == "the page said hello"
                   for message in sent)

    def test_can_itself_ask_for_more_tools(self, fake_ollama):
        """A follow-up turn is a normal turn: it may return tool calls too."""
        call = make_tool_call("fetch_web_page", {"url": "https://example.com"})
        fake_ollama.responses.extend([
            make_chat_response("hi"), make_chat_response("", [call]),
        ])
        module = LLMModule()
        module.chat("bonjour")
        assert module.continue_after_tools()["tool_calls"] is not None

    def test_a_transport_error_is_converted_to_a_message(self, fake_ollama):
        module = LLMModule()
        fake_ollama.error = ConnectionError("ollama is down")
        assert "error" in module.continue_after_tools()["response"].lower()


class TestToolResults:
    def test_tool_result_is_appended_in_protocol_shape(self, fake_ollama):
        module = LLMModule()
        module.add_tool_result("fetch_web_page", "some content")
        assert module.history.get_messages()[-1] == {
            "role": "tool",
            "content": "some content",
            "name": "fetch_web_page",
        }

    def test_reset_drops_the_conversation(self, fake_ollama):
        fake_ollama.responses.append(make_chat_response("hi"))
        module = LLMModule()
        module.chat("hello")
        module.reset_conversation()
        assert all(m["role"] == "system" for m in module.history.get_messages())


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

def test_tool_call_without_content_survives_history_recording(fake_ollama):
    """A tool call with no prose is the normal case, and must not break chat().

    The client returns pydantic models, which are not JSON-serialisable; the
    history normalises them to plain data instead.
    """
    call = make_tool_call("fetch_web_page", {"url": "https://example.com"})
    fake_ollama.responses.append(make_chat_response("", [call]))
    result = LLMModule().chat("read example.com")
    assert result["tool_calls"] == [{
        "function": {
            "name": "fetch_web_page",
            "arguments": {"url": "https://example.com"},
        }
    }]


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


def test_reset_conversation_keeps_exactly_one_system_prompt(fake_ollama):
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
