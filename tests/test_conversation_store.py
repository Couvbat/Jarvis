"""Tests for conversation persistence (conversation_store.py)."""

import pytest

from conversation_store import ConversationStore
from llm_module import ConversationHistory, LLMModule
from tests._stubs import make_chat_response, make_tool_call


@pytest.fixture
def store():
    instance = ConversationStore()
    yield instance
    instance.close()


class TestWriting:
    def test_a_new_conversation_gets_an_id(self, store):
        assert isinstance(store.start(), int)

    def test_ids_are_distinct(self, store):
        assert store.start() != store.start()

    def test_turns_come_back_in_order(self, store):
        conversation = store.start()
        for role, content in [("user", "un"), ("assistant", "deux"), ("user", "trois")]:
            store.append(conversation, {"role": role, "content": content})
        assert [m["content"] for m in store.messages(conversation)] == [
            "un", "deux", "trois",
        ]

    def test_the_role_is_kept(self, store):
        conversation = store.start()
        store.append(conversation, {"role": "assistant", "content": "x"})
        assert store.messages(conversation)[0]["role"] == "assistant"

    def test_a_tool_result_keeps_its_name(self, store):
        conversation = store.start()
        store.append(conversation, {"role": "tool", "content": "x", "name": "fs__read"})
        assert store.messages(conversation)[0]["name"] == "fs__read"

    def test_tool_calls_survive_the_round_trip(self, store):
        conversation = store.start()
        calls = [{"function": {"name": "fs__read", "arguments": {"path": "/tmp/a"}}}]
        store.append(conversation, {"role": "assistant", "content": "", "tool_calls": calls})
        assert store.messages(conversation)[0]["tool_calls"] == calls

    def test_conversations_do_not_mix(self, store):
        first, second = store.start(), store.start()
        store.append(first, {"role": "user", "content": "un"})
        store.append(second, {"role": "user", "content": "deux"})
        assert [m["content"] for m in store.messages(first)] == ["un"]

    def test_a_write_failure_does_not_raise(self, store):
        """Losing the log is not worth losing the conversation."""
        store.close()
        store.append(1, {"role": "user", "content": "x"})


class TestReading:
    def test_a_limit_keeps_the_most_recent_turns(self, store):
        """Resuming wants the end of a conversation, not its beginning."""
        conversation = store.start()
        for index in range(10):
            store.append(conversation, {"role": "user", "content": str(index)})
        assert [m["content"] for m in store.messages(conversation, limit=3)] == [
            "7", "8", "9",
        ]

    def test_an_unknown_conversation_is_empty(self, store):
        assert store.messages(9999) == []

    def test_recent_lists_newest_first(self, store):
        first, second = store.start(), store.start()
        for conversation in (first, second):
            store.append(conversation, {"role": "user", "content": "x"})
        assert [c.id for c in store.recent()] == [second, first]

    def test_recent_counts_messages(self, store):
        conversation = store.start()
        for index in range(3):
            store.append(conversation, {"role": "user", "content": str(index)})
        assert store.recent()[0].message_count == 3

    def test_recent_previews_the_opening_request(self, store):
        conversation = store.start()
        store.append(conversation, {"role": "system", "content": "SYSTEM"})
        store.append(conversation, {"role": "user", "content": "crée un fichier"})
        assert store.recent()[0].preview == "crée un fichier"

    def test_recent_respects_its_limit(self, store):
        for _ in range(5):
            store.append(store.start(), {"role": "user", "content": "x"})
        assert len(store.recent(limit=2)) == 2

    def test_last_id_skips_empty_conversations(self, store):
        """A session that started and said nothing is not worth resuming."""
        used = store.start()
        store.append(used, {"role": "user", "content": "x"})
        store.start()  # opened and abandoned
        assert store.last_id() == used

    def test_last_id_is_none_when_there_is_nothing(self, store):
        assert store.last_id() is None

    def test_search_finds_turns(self, store):
        conversation = store.start()
        store.append(conversation, {"role": "user", "content": "parle-moi du plombier"})
        store.append(conversation, {"role": "user", "content": "et du fromage"})
        assert len(store.search("plombier")) == 1

    def test_search_is_newest_first(self, store):
        conversation = store.start()
        for index in range(3):
            store.append(conversation, {"role": "user", "content": f"note {index}"})
        assert store.search("note")[0]["content"] == "note 2"

    def test_purge_forgets_everything(self, store):
        conversation = store.start()
        store.append(conversation, {"role": "user", "content": "x"})
        store.purge()
        assert store.recent() == []
        assert store.last_id() is None


class TestPersistence:
    def test_conversations_survive_a_restart(self, tmp_path):
        path = tmp_path / "conversations.db"
        first = ConversationStore(path)
        conversation = first.start()
        first.append(conversation, {"role": "user", "content": "bonjour"})
        first.close()

        second = ConversationStore(path)
        assert [m["content"] for m in second.messages(conversation)] == ["bonjour"]
        second.close()

    def test_the_parent_directory_is_created(self, tmp_path):
        store = ConversationStore(tmp_path / "nested" / "dir" / "conversations.db")
        store.start()
        assert (tmp_path / "nested" / "dir" / "conversations.db").exists()
        store.close()

    def test_ending_a_conversation_is_recorded(self, store):
        conversation = store.start()
        store.append(conversation, {"role": "user", "content": "x"})
        store.end(conversation)
        assert store.recent()[0].ended_at is not None


class TestHistoryIntegration:
    def test_every_turn_is_recorded(self, store):
        conversation = store.start()
        history = ConversationHistory(
            max_history=10, on_turn=lambda m: store.append(conversation, m)
        )
        history.add_user("bonjour")
        history.add_assistant("salut")
        assert len(store.messages(conversation)) == 2

    def test_trimming_does_not_erase_the_record(self, store):
        """What is dropped to fit the context window is still what was said."""
        conversation = store.start()
        history = ConversationHistory(
            max_history=2, on_turn=lambda m: store.append(conversation, m)
        )
        for index in range(6):
            history.add_user(str(index))
        assert len(history.get_messages()) == 2
        assert len(store.messages(conversation)) == 6

    def test_a_failing_recorder_does_not_break_the_conversation(self):
        def explode(message):
            raise RuntimeError("disk full")

        history = ConversationHistory(max_history=10, on_turn=explode)
        history.add_user("bonjour")
        assert len(history.get_messages()) == 1

    def test_restoring_does_not_re_record(self, store):
        conversation = store.start()
        history = ConversationHistory(
            max_history=10, on_turn=lambda m: store.append(conversation, m)
        )
        history.restore([
            {"role": "user", "content": "ancien"},
            {"role": "assistant", "content": "ancienne réponse"},
        ])
        assert len(history.get_messages()) == 2
        assert store.messages(conversation) == []

    def test_restoring_skips_the_system_prompt(self, store):
        history = ConversationHistory(max_history=10)
        history.set_system("SYSTEM")
        history.restore([
            {"role": "system", "content": "OLD SYSTEM"},
            {"role": "user", "content": "bonjour"},
        ])
        messages = history.get_messages()
        assert messages[0]["content"] == "SYSTEM"
        assert len(messages) == 2


class TestModuleIntegration:
    async def test_a_turn_is_persisted(self, store, fake_ollama):
        fake_ollama.responses.append(make_chat_response("Bonjour"))
        module = LLMModule(store=store)
        await module.chat("salut")

        recorded = store.messages(module.conversation_id)
        assert [m["role"] for m in recorded] == ["user", "assistant"]

    async def test_tool_calls_and_results_are_persisted(self, store, fake_ollama):
        call = make_tool_call("fs__read", {"path": "/tmp/a"})
        fake_ollama.responses.append(make_chat_response("", [call]))
        module = LLMModule(store=store)
        await module.chat("lis le fichier")
        module.add_tool_result("fs__read", "le contenu")

        recorded = store.messages(module.conversation_id)
        assert recorded[1]["tool_calls"][0]["function"]["name"] == "fs__read"
        assert recorded[2]["role"] == "tool"

    async def test_resuming_brings_back_the_last_conversation(self, store, fake_ollama):
        fake_ollama.responses.append(make_chat_response("Bonjour"))
        first = LLMModule(store=store)
        await first.chat("parle-moi du plombier")
        first.close()

        second = LLMModule(store=store)
        assert second.resume() == 2
        assert any(
            "plombier" in message["content"]
            for message in second.history.get_messages()
        )

    async def test_resuming_keeps_only_the_recent_turns(self, store, fake_ollama, settings):
        settings.max_conversation_history = 4
        fake_ollama.responses.extend([make_chat_response(f"r{i}") for i in range(6)])
        first = LLMModule(store=store)
        for index in range(6):
            await first.chat(f"question {index}")
        first.close()

        second = LLMModule(store=store)
        assert second.resume() == 4

    def test_resuming_with_nothing_stored_is_a_no_op(self, store):
        assert LLMModule(store=store).resume() == 0

    def test_resuming_never_reloads_the_current_conversation(self, store):
        module = LLMModule(store=store)
        module.history.add_user("dans cette session")
        assert module.resume(module.conversation_id) == 0

    def test_without_a_store_nothing_is_recorded(self):
        module = LLMModule()
        assert module.conversation_id is None
        module.resume()  # must not raise
        module.close()

    async def test_resetting_opens_a_new_conversation(self, store, fake_ollama):
        """What was said was said; a reset does not rewrite the log."""
        fake_ollama.responses.append(make_chat_response("Bonjour"))
        module = LLMModule(store=store)
        await module.chat("salut")
        first = module.conversation_id

        module.reset_conversation()
        assert module.conversation_id != first
        assert len(store.messages(first)) == 2
