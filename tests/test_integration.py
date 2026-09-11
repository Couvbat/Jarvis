"""End-to-end flows wiring the real modules together.

Only the process boundaries are faked (the Ollama server, the microphone, the
Piper binary); ``LLMModule``, ``ActionExecutor``, ``WhitelistManager`` and
``Jarvis`` are the real implementations.
"""

import numpy as np
import pytest

import main as main_module
from action_executor import ActionExecutor
from llm_module import LLMModule
from main import Jarvis
from tests._stubs import make_chat_response, make_tool_call


@pytest.fixture
def wired(monkeypatch, sandbox, tmp_path, fake_ollama):
    """A Jarvis whose LLM and executor are real, with fake audio at the edges."""
    monkeypatch.chdir(tmp_path)

    class FakeAudio:
        sample_rate = 16000

        def __init__(self):
            self.played = []

        def record_until_silence(self, **kwargs):
            return np.zeros(16000, dtype=np.int16)

        def play_audio(self, audio, sample_rate=None):
            self.played.append((audio, sample_rate))

    class FakeSTT:
        def __init__(self):
            self.script = []
            self.language = "en"

        def initialize(self):
            pass

        def set_language(self, language):
            self.language = language

        def transcribe(self, audio, sample_rate=16000):
            return self.script.pop(0) if self.script else "exit"

    class FakeTTS:
        def __init__(self):
            self.spoken = []

        def initialize(self):
            pass

        def synthesize(self, text):
            self.spoken.append(text)
            return np.zeros(2205, dtype=np.int16)

    audio, stt, tts = FakeAudio(), FakeSTT(), FakeTTS()
    monkeypatch.setattr(main_module, "AudioHandler", lambda: audio)
    monkeypatch.setattr(main_module, "STTModule", lambda: stt)
    monkeypatch.setattr(main_module, "TTSModule", lambda: tts)

    approvals = []

    def always_approve(description, item):
        approvals.append((description, item))
        return (True, False)

    instance = Jarvis(use_tui=False)
    instance.executor.confirmation_callback = always_approve
    instance._approvals = approvals
    instance._audio, instance._stt, instance._tts = audio, stt, tts
    return instance


class TestFileCreationFlow:
    def test_voice_request_creates_a_file(self, wired, sandbox, fake_ollama):
        target = sandbox / "notes.txt"
        fake_ollama.responses.extend([
            make_chat_response("I'll create it", [make_tool_call(
                "execute_file_operation",
                {"operation": "create_file", "path": str(target),
                 "content": "ma liste"},
            )]),
            make_chat_response("C'est fait."),
        ])
        wired._stt.script = ["crée un fichier notes.txt", "exit"]
        wired.run_interactive()

        assert target.read_text() == "ma liste"
        assert "C'est fait." in wired._tts.spoken

    def test_the_user_is_asked_before_writing(self, wired, sandbox, fake_ollama):
        target = sandbox / "notes.txt"
        fake_ollama.responses.extend([
            make_chat_response("ok", [make_tool_call(
                "execute_file_operation",
                {"operation": "create_file", "path": str(target), "content": "x"},
            )]),
            make_chat_response("done"),
        ])
        wired._stt.script = ["crée un fichier", "exit"]
        wired.run_interactive()
        assert wired._approvals

    def test_a_refused_action_writes_nothing(self, wired, sandbox, fake_ollama):
        target = sandbox / "notes.txt"
        wired.executor.confirmation_callback = lambda d, i: (False, False)
        fake_ollama.responses.extend([
            make_chat_response("ok", [make_tool_call(
                "execute_file_operation",
                {"operation": "create_file", "path": str(target), "content": "x"},
            )]),
            make_chat_response("Annulé."),
        ])
        wired._stt.script = ["crée un fichier", "exit"]
        wired.run_interactive()
        assert not target.exists()

    def test_a_path_outside_the_sandbox_is_refused_end_to_end(
        self, wired, tmp_path, fake_ollama
    ):
        target = tmp_path / "escaped.txt"
        fake_ollama.responses.extend([
            make_chat_response("ok", [make_tool_call(
                "execute_file_operation",
                {"operation": "create_file", "path": str(target), "content": "x"},
            )]),
            make_chat_response("Je ne peux pas."),
        ])
        wired._stt.script = ["écris dans /etc", "exit"]
        wired.run_interactive()
        assert not target.exists()
        assert wired._approvals == []


class TestToolResultFeedback:
    def test_the_tool_result_reaches_the_next_prompt(self, wired, sandbox, fake_ollama):
        (sandbox / "notes.txt").write_text("le contenu du fichier")
        fake_ollama.responses.extend([
            make_chat_response("ok", [make_tool_call(
                "execute_file_operation",
                {"operation": "read_file", "path": str(sandbox / "notes.txt")},
            )]),
            make_chat_response("Le fichier dit: le contenu du fichier"),
        ])
        wired._stt.script = ["lis notes.txt", "exit"]
        wired.run_interactive()

        follow_up_messages = fake_ollama.calls[-1]["messages"]
        assert any("le contenu du fichier" in message["content"]
                   for message in follow_up_messages if message["role"] == "tool")


class TestWhitelistPersistence:
    def test_approving_once_skips_the_second_prompt(self, sandbox, tmp_path,
                                                    monkeypatch, fake_ollama):
        monkeypatch.chdir(tmp_path)
        prompts = []

        def approve_and_remember(description, item):
            prompts.append(item)
            return (True, True)

        executor = ActionExecutor(confirmation_callback=approve_and_remember)
        executor.execute_file_operation("create_file", str(sandbox / "a.txt"), "a")
        executor.execute_file_operation("create_file", str(sandbox / "b.txt"), "b")

        # A fresh executor reads the same on-disk whitelist.
        reloaded = ActionExecutor(confirmation_callback=approve_and_remember)
        reloaded.execute_file_operation("create_file", str(sandbox / "c.txt"), "c")

        assert len(prompts) == 1
        assert (sandbox / "c.txt").exists()
        assert (tmp_path / "command_whitelist.json").exists()


class TestLanguageFlow:
    def test_switching_language_persists_across_turns(self, wired, fake_ollama):
        fake_ollama.responses.append(make_chat_response("Bonjour"))
        wired._stt.script = ["switch to french", "bonjour", "exit"]
        wired.run_interactive()
        assert wired._stt.language == "fr"

    def test_the_llm_never_sees_the_switch_command(self, wired, fake_ollama):
        fake_ollama.responses.append(make_chat_response("Bonjour"))
        wired._stt.script = ["switch to french", "bonjour", "exit"]
        wired.run_interactive()
        user_turns = [
            message["content"]
            for call in fake_ollama.calls
            for message in call["messages"]
            if message["role"] == "user"
        ]
        assert "switch to french" not in user_turns


class TestDegradedServices:
    def test_a_dead_ollama_still_answers_the_user(self, wired, fake_ollama):
        fake_ollama.error = ConnectionError("connection refused")
        wired._stt.script = ["bonjour", "exit"]
        wired.run_interactive()
        assert any("error" in text.lower() for text in wired._tts.spoken)

    def test_a_dead_ollama_does_not_end_the_session(self, wired, fake_ollama):
        fake_ollama.error = ConnectionError("connection refused")
        wired._stt.script = ["bonjour", "encore", "exit"]
        wired.run_interactive()
        assert len(wired._stt.script) == 0


class TestSystemPromptContract:
    def test_every_capability_named_in_the_prompt_has_a_tool(self):
        prompt = LLMModule.SYSTEM_PROMPT.lower()
        names = {tool["function"]["name"] for tool in LLMModule.TOOLS}
        assert "file operations" in prompt and "execute_file_operation" in names
        assert "web" in prompt and "fetch_web_page" in names
        assert "application" in prompt and "launch_application" in names


def test_file_creation_works_with_a_legacy_dict_response(wired, sandbox, fake_ollama):
    """Older ollama clients hand back plain dicts; both shapes must work."""
    target = sandbox / "legacy.txt"
    fake_ollama.responses.extend([
        make_chat_response("", [make_tool_call(
            "execute_file_operation",
            {"operation": "create_file", "path": str(target), "content": "ok"},
            as_model=False,
        )], as_model=False),
        make_chat_response("C'est fait.", as_model=False),
    ])
    wired._stt.script = ["crée un fichier", "exit"]
    wired.run_interactive()
    assert target.read_text() == "ok"
