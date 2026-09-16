"""Tests for the orchestration loop (main.py)."""

import asyncio
import builtins

import numpy as np
import pytest

import main as main_module
from main import Jarvis, is_exit_command, normalise_utterance
from policy.engine import Decision, Surface
from policy.store import ApprovalStore
from tools.registry import ToolRegistry
from tools.schema import Risk, ToolResult, ToolSpec


class FakeAudio:
    sample_rate = 16000

    def __init__(self):
        self.played = []

    def record_until_silence(self, **kwargs):
        return np.zeros(16000, dtype=np.int16)

    def play_audio(self, audio, sample_rate=None):
        self.played.append((audio, sample_rate))


class FakeSTT:
    def __init__(self, script=None):
        self.script = list(script or [])
        self.language = "en"
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def set_language(self, language):
        self.language = language

    def transcribe(self, audio, sample_rate=16000):
        return self.script.pop(0) if self.script else "exit"


class FakeLLM:
    MAX_TOOL_ITERATIONS = 5

    conversation_id = None
    available = True
    model_pulled = True

    async def is_available(self):
        return self.available

    async def has_model(self):
        return self.model_pulled

    #: detail line for a second, configured-but-not-serving provider
    fallback_detail = None

    async def status(self):
        """Mirrors LLMModule.status(): the provider that will answer, if any,
        and what every configured provider reported."""
        from llm_providers import ProviderConfig

        report = {}
        if not self.available:
            report["primary"] = "unreachable: ConnectionError"
        elif not self.model_pulled:
            report["primary"] = "reachable, but 'llama3.1:8b' is not pulled"
        else:
            report["primary"] = "ready"
        if self.fallback_detail is not None:
            report["fallback"] = self.fallback_detail

        if report["primary"] != "ready":
            return None, report
        return (
            ProviderConfig(
                name="primary", host="http://localhost:11434", model="llama3.1:8b"
            ),
            report,
        )

    def resume(self, conversation_id=None):
        return 0

    def close(self):
        pass

    def __init__(self, script=None):
        self.script = list(script or [])
        self.chats = []          # user messages passed to chat()
        self.follow_ups = 0      # continue_after_tools() calls
        self.tool_results = []
        self.untrusted_results = []

    def _next(self):
        if self.script:
            return self.script.pop(0)
        return {"response": "ok", "tool_calls": None}

    async def chat(self, message, on_text=None):
        self.chats.append(message)
        return self._stream(self._next(), on_text)

    async def continue_after_tools(self, on_text=None):
        self.follow_ups += 1
        return self._stream(self._next(), on_text)

    @staticmethod
    def _stream(result, on_text):
        """Hand the text over in slices, as the real client does."""
        text = result.get("response") or ""
        if on_text and text:
            for start in range(0, len(text), 5):
                on_text(text[start:start + 5])
        return result

    def add_tool_result(self, name, result, untrusted=False):
        self.tool_results.append((name, result))
        if untrusted:
            self.untrusted_results.append(name)


class FakeTTS:
    def __init__(self):
        self.spoken = []
        self.initialized = False

    def initialize(self):
        self.initialized = True

    #: rate of the loaded voice; ``*-low`` Piper voices are 16 kHz, not 22050
    sample_rate = 16000

    def synthesize(self, text):
        self.spoken.append(text)
        return np.zeros(2205, dtype=np.int16), self.sample_rate


def build_test_registry(executed, risk=Risk.READ_ONLY):
    """A real registry holding recording tools, so dispatch is not faked."""
    def record(**kwargs):
        executed.append(kwargs)
        return ToolResult("tool ok")

    registry = ToolRegistry()
    for name in ("fs__write", "web__fetch", "fs__read"):
        registry.register(ToolSpec(
            name=name,
            description=f"{name} does something",
            input_schema={"type": "object", "properties": {}},
            handler=record,
            risk=risk,
        ))
    return registry


@pytest.fixture
def wiring(monkeypatch):
    """Replace the process boundaries; keep the registry and policy real."""
    executed = []
    registry = build_test_registry(executed)
    parts = {
        "audio": FakeAudio(),
        "stt": FakeSTT(),
        "llm": FakeLLM(),
        "tts": FakeTTS(),
        "registry": registry,
        "executed": executed,
    }
    monkeypatch.setattr(main_module, "AudioHandler", lambda: parts["audio"])
    monkeypatch.setattr(main_module, "STTModule", lambda: parts["stt"])
    monkeypatch.setattr(
        main_module, "LLMModule", lambda *args, **kwargs: parts["llm"]
    )
    monkeypatch.setattr(main_module, "TTSModule", lambda: parts["tts"])
    monkeypatch.setattr(main_module, "build_default_registry", lambda: registry)
    monkeypatch.setattr(main_module, "ApprovalStore", lambda path: ApprovalStore(":memory:"))
    return parts


@pytest.fixture
async def jarvis(wiring):
    """A Jarvis that approves every confirmation, so tests can focus on flow."""
    instance = Jarvis(use_tui=False)
    instance._confirm = lambda decision: (True, False)
    await instance.speech.start()
    yield instance
    await instance.speech.stop()


class TestUtteranceNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("Stop.", "stop"),
        ("Au revoir !", "au revoir"),
        ("ARRÊTE", "arrete"),
        ("Arrête-toi", "arrete toi"),
        ("  spaced   out  ", "spaced out"),
        ("s'il te plaît", "s il te plait"),
        ("Crée un fichier café.txt", "cree un fichier cafe txt"),
    ])
    def test_normalisation(self, raw, expected):
        assert normalise_utterance(raw) == expected


class TestExitCommand:
    @pytest.mark.parametrize("phrase", [
        "exit", "quit", "stop", "goodbye", "bye",
        "au revoir", "arrête", "arrête-toi",
    ])
    def test_bare_commands_quit(self, phrase):
        assert is_exit_command(phrase) is True

    @pytest.mark.parametrize("phrase", [
        "Stop.", "STOP", "Au revoir !", "  quit  ",
    ])
    def test_case_accents_and_punctuation_are_tolerated(self, phrase):
        assert is_exit_command(phrase) is True

    @pytest.mark.parametrize("phrase", [
        "Jarvis, stop", "stop jarvis", "ok quit",
        "stop please", "arrête s'il te plaît", "quit now",
        "Jarvis, stop please",
    ])
    def test_address_and_politeness_are_stripped(self, phrase):
        assert is_exit_command(phrase) is True

    @pytest.mark.parametrize("phrase", [
        "stop the music please",
        "arrête la musique",
        "don't stop the recording",
        "je ne veux pas quitter",
        "goodbye everyone, now write that to a file",
        "how do I exit vim",
        "crée un fichier stop.txt",
        "",
        "   ",
    ])
    def test_a_sentence_merely_containing_the_word_does_not_quit(self, phrase):
        """This is the whole point: "stop" is a common word in real requests."""
        assert is_exit_command(phrase) is False


class TestStartup:
    async def test_models_load_for_a_voice_session(self, jarvis, wiring):
        await jarvis.start()
        assert wiring["stt"].initialized is True
        assert wiring["tts"].initialized is True

    async def test_a_text_session_loads_neither(self, jarvis, wiring):
        """--text is documented as working without audio; it used to download
        and load Whisper and Piper before the first prompt."""
        jarvis.speak_aloud = False
        await jarvis.start()
        assert wiring["stt"].initialized is False
        assert wiring["tts"].initialized is False

    def test_tools_and_policy_are_assembled(self, jarvis, wiring):
        assert jarvis.registry is wiring["registry"]
        assert jarvis.policy is not None
        assert jarvis.taint.tainted is False

    def test_no_tui_by_default(self, jarvis):
        assert jarvis.use_tui is False
        assert jarvis.tui is None


def make_decision(surface=Surface.VOICE, risk=Risk.WRITE):
    return Decision(
        tool="fs__write", risk=risk, scope="/tmp",
        surface=surface, summary="fs__write (path=/tmp/a.txt)",
        reason="write access",
    )


class TestConfirmationPrompt:
    @pytest.fixture
    def asking(self, wiring):
        """A Jarvis with the real _confirm, not the auto-approving stub."""
        return Jarvis(use_tui=False)

    @pytest.mark.parametrize("answer,expected", [
        ("y", (True, False)),
        ("a", (True, True)),
        ("n", (False, False)),
        ("Y", (True, False)),
        (" a ", (True, True)),
    ])
    def test_answers(self, asking, monkeypatch, answer, expected):
        monkeypatch.setattr(builtins, "input", lambda *a: answer)
        assert asking._confirm(make_decision()) == expected

    def test_invalid_answers_reprompt(self, asking, monkeypatch):
        answers = iter(["maybe", "", "y"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        assert asking._confirm(make_decision()) == (True, False)

    def test_the_real_arguments_are_shown(self, asking, monkeypatch, capsys):
        monkeypatch.setattr(builtins, "input", lambda *a: "n")
        asking._confirm(make_decision())
        output = capsys.readouterr().out
        assert "path=/tmp/a.txt" in output
        assert "write access" in output

    def test_a_terminal_decision_does_not_offer_always(self, asking, monkeypatch, capsys):
        """Destructive and tainted calls should be seen every time."""
        monkeypatch.setattr(builtins, "input", lambda *a: "n")
        asking._confirm(make_decision(Surface.TERMINAL, Risk.DESTRUCTIVE))
        output = capsys.readouterr().out
        assert "stop asking" not in output
        assert "keyboard" in output

    def test_always_is_refused_on_a_terminal_decision(self, asking, monkeypatch):
        answers = iter(["a", "y"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        assert asking._confirm(make_decision(Surface.TERMINAL)) == (True, False)


class TestProcessUserInput:
    async def test_plain_response_is_returned(self, jarvis, wiring):
        wiring["llm"].script = [{"response": "Bonjour", "tool_calls": None}]
        assert (await jarvis.process_user_input("salut")) == "Bonjour"

    async def test_no_tools_means_a_single_llm_round(self, jarvis, wiring):
        wiring["llm"].script = [{"response": "Bonjour", "tool_calls": None}]
        await jarvis.process_user_input("salut")
        assert len(wiring["llm"].chats) == 1

    async def test_tool_calls_are_executed(self, jarvis, wiring):
        call = {"function": {"name": "web__fetch", "arguments": {"url": "u"}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "Voici le résumé", "tool_calls": None},
        ]
        assert (await jarvis.process_user_input("lis le site")) == "Voici le résumé"
        assert wiring["executed"] == [{"url": "u"}]

    async def test_tool_results_are_fed_back(self, jarvis, wiring):
        call = {"function": {"name": "web__fetch", "arguments": {"url": "u"}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        await jarvis.process_user_input("lis le site")
        assert wiring["llm"].tool_results == [("web__fetch", "tool ok")]

    async def test_a_second_round_of_tool_calls_also_runs(self, jarvis, wiring):
        """The follow-up turn may ask for more tools; they must be executed."""
        first = {"function": {"name": "fs__write", "arguments": {"n": 1}}}
        second = {"function": {"name": "web__fetch", "arguments": {"n": 2}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [first]},
            {"response": "", "tool_calls": [second]},
            {"response": "fini", "tool_calls": None},
        ]
        assert (await jarvis.process_user_input("fais les deux")) == "fini"
        assert wiring["executed"] == [{"n": 1}, {"n": 2}]

    async def test_the_tool_loop_is_bounded(self, jarvis, wiring):
        """A model that keeps asking for tools must not loop forever."""
        call = {"function": {"name": "web__fetch", "arguments": {}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]} for _ in range(50)
        ]
        await jarvis.process_user_input("boucle")
        assert len(wiring["executed"]) == FakeLLM.MAX_TOOL_ITERATIONS

    async def test_the_follow_up_is_not_a_user_turn(self, jarvis, wiring):
        call = {"function": {"name": "web__fetch", "arguments": {}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        await jarvis.process_user_input("lis le site")
        assert wiring["llm"].chats == ["lis le site"]
        assert wiring["llm"].follow_ups == 1

    async def test_several_tool_calls_all_run(self, jarvis, wiring):
        calls = [
            {"function": {"name": "fs__write", "arguments": {}}},
            {"function": {"name": "web__fetch", "arguments": {}}},
        ]
        wiring["llm"].script = [
            {"response": "", "tool_calls": calls},
            {"response": "done", "tool_calls": None},
        ]
        await jarvis.process_user_input("fais deux choses")
        assert len(wiring["executed"]) == 2


class TestEventLoopIsNotBlocked:
    """Blocking work must run off the loop, or Phase 3's barge-in cannot hear
    the user while Jarvis is speaking."""

    async def test_recording_runs_in_a_thread(self, jarvis, wiring, monkeypatch):
        import threading

        loop_thread = threading.get_ident()
        seen = {}

        def record(**kwargs):
            seen["thread"] = threading.get_ident()
            return np.zeros(16000, dtype=np.int16)

        monkeypatch.setattr(wiring["audio"], "record_until_silence", record)
        wiring["stt"].script = ["exit"]
        await jarvis.run_interactive()
        assert seen["thread"] != loop_thread

    async def test_transcription_runs_in_a_thread(self, jarvis, wiring, monkeypatch):
        import threading

        loop_thread = threading.get_ident()
        seen = {}

        def transcribe(audio, sample_rate=16000):
            seen["thread"] = threading.get_ident()
            return "exit"

        monkeypatch.setattr(wiring["stt"], "transcribe", transcribe)
        await jarvis.run_interactive()
        assert seen["thread"] != loop_thread

    async def test_synthesis_runs_in_a_thread(self, jarvis, wiring, monkeypatch):
        import threading

        loop_thread = threading.get_ident()
        seen = {}

        def synthesize(text):
            seen["thread"] = threading.get_ident()
            return np.zeros(2205, dtype=np.int16), 16000

        monkeypatch.setattr(wiring["tts"], "synthesize", synthesize)
        wiring["stt"].script = ["exit"]
        await jarvis.run_interactive()
        assert seen["thread"] != loop_thread

    async def test_the_confirmation_prompt_runs_in_a_thread(self, wiring, monkeypatch):
        """It reads stdin; a pending question must not freeze everything else."""
        import threading

        instance = Jarvis(use_tui=False)
        loop_thread = threading.get_ident()
        seen = {}

        def confirm(decision):
            seen["thread"] = threading.get_ident()
            return (True, False)

        instance._confirm = confirm
        call = {"function": {"name": "fs__write", "arguments": {}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        await instance.process_user_input("écris")
        assert seen["thread"] != loop_thread


class TestBargeIn:
    """Speaking over an answer ends it and starts the next turn."""

    async def test_disabled_by_default_the_answer_finishes(self, jarvis, wiring, settings):
        """On open speakers the microphone hears Jarvis, so this is opt-in."""
        assert settings.barge_in is False
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Une réponse complète.", "tool_calls": None}]
        await jarvis.run_interactive()
        assert any("réponse" in text for text in wiring["tts"].spoken)

    async def test_an_interruption_cuts_the_answer_short(
        self, jarvis, wiring, settings, monkeypatch
    ):
        settings.barge_in = True
        interrupted = []

        class Interrupting:
            def __init__(self, audio, **kwargs):
                self.detected = asyncio.Event()
                self.detected.set()  # the user was already talking

            async def start(self):
                pass

            async def stop(self):
                return np.full(320, 1234, dtype=np.int16)

        monkeypatch.setattr(main_module, "BargeInListener", Interrupting)
        monkeypatch.setattr(
            jarvis.speech, "interrupt",
            lambda: interrupted.append(True) or asyncio.sleep(0),
        )

        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Une réponse.", "tool_calls": None}]
        await jarvis.run_interactive()
        assert interrupted

    async def test_the_interrupting_words_start_the_next_turn(
        self, jarvis, wiring, settings, monkeypatch
    ):
        """Losing the start of a sentence would make interrupting worse than
        waiting."""
        settings.barge_in = True
        captured = np.full(320, 1234, dtype=np.int16)

        class Interrupting:
            def __init__(self, audio, **kwargs):
                self.detected = asyncio.Event()
                self.detected.set()

            async def start(self):
                pass

            async def stop(self):
                return captured

        monkeypatch.setattr(main_module, "BargeInListener", Interrupting)

        prefixes = []

        def record(**kwargs):
            prefixes.append(kwargs.get("prefix"))
            return np.zeros(16000, dtype=np.int16)

        monkeypatch.setattr(wiring["audio"], "record_until_silence", record)
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Une réponse.", "tool_calls": None}]
        await jarvis.run_interactive()

        assert any(prefix is not None for prefix in prefixes)

    async def test_the_carried_audio_is_used_once(self, jarvis, wiring, settings):
        """It belongs to the turn it started, not to every turn after."""
        jarvis._carried_audio = np.full(320, 1234, dtype=np.int16)
        assert jarvis._carried_audio is not None
        wiring["stt"].script = ["exit"]
        await jarvis.run_interactive()
        assert jarvis._carried_audio is None

    async def test_text_mode_never_listens_for_an_interruption(
        self, jarvis, wiring, settings, monkeypatch
    ):
        settings.barge_in = True
        built = []
        monkeypatch.setattr(
            main_module, "BargeInListener", lambda *a, **k: built.append(True)
        )
        answers = iter(["bonjour", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        await jarvis.run_text_mode()
        assert built == []


class TestLanguageSwitching:
    @pytest.mark.parametrize("phrase", [
        "switch to french", "parle français", "en français", "Parle Français !",
    ])
    async def test_switching_to_french(self, jarvis, wiring, phrase):
        assert await jarvis.handle_language_switch(phrase) is True
        assert wiring["stt"].language == "fr"

    @pytest.mark.parametrize("phrase", [
        "switch to english", "parle anglais", "in english",
    ])
    async def test_switching_to_english(self, jarvis, wiring, phrase):
        wiring["stt"].language = "fr"
        assert await jarvis.handle_language_switch(phrase) is True
        assert wiring["stt"].language == "en"

    async def test_an_ordinary_request_is_not_a_switch(self, jarvis, wiring):
        assert await jarvis.handle_language_switch("crée un fichier") is False
        assert wiring["llm"].chats == []

    async def test_the_switch_is_confirmed_aloud(self, jarvis, wiring):
        await jarvis.handle_language_switch("parle français")
        assert any("français" in text for text in wiring["tts"].spoken)


class TestOllamaHealth:
    async def test_a_reachable_server_says_nothing(self, jarvis, wiring, capsys):
        await jarvis._check_llm()
        assert "No LLM provider" not in capsys.readouterr().out

    async def test_an_unreachable_server_is_reported_once(self, jarvis, wiring, capsys):
        """Otherwise a stopped Ollama is just a spoken apology per turn, which
        tells the user nothing about what to fix."""
        wiring["llm"].available = False
        await jarvis._check_llm()
        output = capsys.readouterr().out
        assert "No LLM provider is usable" in output
        assert "unreachable" in output
        assert "ollama serve" in output

    async def test_a_missing_model_is_reported(self, jarvis, wiring, capsys):
        """A server that is up without its model cannot answer either, and the
        fix is a different command."""
        wiring["llm"].model_pulled = False
        await jarvis._check_llm()
        output = capsys.readouterr().out
        assert "not pulled" in output
        assert "ollama pull" in output

    async def test_the_serving_provider_is_named_in_the_tui(self, tui_jarvis, wiring):
        """With two providers configured, which model is answering is the
        thing worth knowing at startup."""
        await tui_jarvis._check_llm()
        messages = [
            payload for name, payload in tui_jarvis._tui.events
            if name == "add_system_message"
        ]
        assert any("llama3.1:8b" in str(payload) for payload in messages)

    async def test_the_other_providers_are_listed_too(self, tui_jarvis, wiring):
        """"Falling back" is only useful information if you can see what the
        preferred provider is doing instead."""
        wiring["llm"].fallback_detail = "unreachable: ConnectionError"
        await tui_jarvis._check_llm()
        messages = [
            str(payload) for name, payload in tui_jarvis._tui.events
            if name == "add_system_message"
        ]
        assert any("fallback" in text and "unreachable" in text
                   for text in messages)

    async def test_a_failure_reaches_the_tui_too(self, tui_jarvis, wiring):
        wiring["llm"].available = False
        await tui_jarvis._check_llm()
        messages = [
            str(payload) for name, payload in tui_jarvis._tui.events
            if name == "add_system_message"
        ]
        assert any("No LLM provider is usable" in text for text in messages)

    async def test_startup_checks_the_server(self, jarvis, wiring, monkeypatch):
        checked = []
        monkeypatch.setattr(
            jarvis, "_check_llm", lambda: checked.append(True) or asyncio.sleep(0)
        )
        await jarvis.start()
        assert checked


class TestVoiceLoop:
    async def test_exit_command_ends_the_loop(self, jarvis, wiring):
        wiring["stt"].script = ["exit"]
        await jarvis.run_interactive()
        assert wiring["tts"].spoken[-1] in ("Goodbye!", "Au revoir!")

    @pytest.mark.parametrize("phrase", ["exit", "quit", "goodbye", "au revoir"])
    async def test_exit_synonyms(self, wiring, phrase):
        wiring["stt"].script = [phrase]
        await Jarvis(use_tui=False).run_interactive()
        assert wiring["tts"].spoken

    async def test_a_normal_turn_is_spoken_then_the_loop_continues(self, jarvis, wiring):
        wiring["stt"].script = ["quelle heure est-il", "exit"]
        wiring["llm"].script = [{"response": "Il est midi", "tool_calls": None}]
        await jarvis.run_interactive()
        assert "Il est midi" in wiring["tts"].spoken

    async def test_switch_to_french(self, jarvis, wiring):
        wiring["stt"].script = ["switch to french", "exit"]
        await jarvis.run_interactive()
        assert wiring["stt"].language == "fr"

    async def test_switch_to_english(self, jarvis, wiring):
        wiring["stt"].script = ["parle anglais", "exit"]
        await jarvis.run_interactive()
        assert wiring["stt"].language == "en"

    async def test_language_switch_does_not_reach_the_llm(self, jarvis, wiring):
        wiring["stt"].script = ["en français", "exit"]
        await jarvis.run_interactive()
        assert wiring["llm"].chats == []

    async def test_empty_transcription_is_skipped(self, jarvis, wiring):
        wiring["stt"].script = ["", "exit"]
        await jarvis.run_interactive()
        assert wiring["llm"].chats == []

    async def test_short_recording_is_skipped(self, jarvis, wiring, monkeypatch):
        # First pass is too short to transcribe; the second one ends the loop.
        buffers = iter([np.zeros(10, dtype=np.int16), np.zeros(16000, dtype=np.int16)])
        monkeypatch.setattr(
            wiring["audio"], "record_until_silence", lambda **kw: next(buffers)
        )
        transcribed = []

        def counting_transcribe(*args, **kwargs):
            transcribed.append(1)
            return "exit"

        monkeypatch.setattr(wiring["stt"], "transcribe", counting_transcribe)
        await jarvis.run_interactive()
        assert len(transcribed) == 1

    async def test_transcription_error_does_not_kill_the_loop(self, jarvis, wiring, monkeypatch):
        results = iter([RuntimeError("stt down"), "exit"])

        def flaky(*args, **kwargs):
            value = next(results)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(wiring["stt"], "transcribe", flaky)
        await jarvis.run_interactive()
        assert wiring["tts"].spoken

    async def test_empty_synthesis_falls_back_to_text(self, jarvis, wiring, monkeypatch, capsys):
        """Nothing to play means say it in the terminal, not stay silent."""
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
        monkeypatch.setattr(
            wiring["tts"], "synthesize",
            lambda text: (np.array([], dtype=np.int16), 16000),
        )
        await jarvis.run_interactive()
        assert "Salut" in capsys.readouterr().out

    async def test_tts_failure_falls_back_to_text(self, jarvis, wiring, monkeypatch, capsys):
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]

        def broken(text):
            raise RuntimeError("piper missing")

        monkeypatch.setattr(wiring["tts"], "synthesize", broken)
        await jarvis.run_interactive()
        assert "Salut" in capsys.readouterr().out

    async def test_processing_error_is_reported_not_raised(self, jarvis, wiring, monkeypatch):
        wiring["stt"].script = ["bonjour", "exit"]

        def broken(text):
            raise RuntimeError("llm down")

        monkeypatch.setattr(jarvis, "process_user_input", broken)
        await jarvis.run_interactive()
        assert any("error" in text.lower() for text in wiring["tts"].spoken)


class TestTextMode:
    async def test_processes_input_and_exits(self, jarvis, wiring, monkeypatch, capsys):
        answers = iter(["bonjour", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
        await jarvis.run_text_mode()
        assert "Salut" in capsys.readouterr().out

    async def test_blank_lines_are_ignored(self, jarvis, wiring, monkeypatch):
        answers = iter(["", "   ", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        await jarvis.run_text_mode()
        assert wiring["llm"].chats == []

    async def test_ctrl_c_exits_cleanly(self, jarvis, wiring, monkeypatch):
        def interrupt(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr(builtins, "input", interrupt)
        await jarvis.run_text_mode()

    async def test_errors_are_reported_not_raised(self, jarvis, wiring, monkeypatch, capsys):
        answers = iter(["bonjour", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))

        def broken(text):
            raise RuntimeError("llm down")

        monkeypatch.setattr(jarvis, "process_user_input", broken)
        await jarvis.run_text_mode()
        assert "error" in capsys.readouterr().out.lower()

    async def test_no_audio_is_played(self, jarvis, wiring, monkeypatch):
        answers = iter(["bonjour", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        await jarvis.run_text_mode()
        assert wiring["audio"].played == []


class TestEntryPoint:
    @pytest.fixture(autouse=True)
    def isolated_cwd(self, tmp_path, monkeypatch):
        """``main()`` configures a file logger; keep jarvis.log out of the repo."""
        monkeypatch.chdir(tmp_path)

    def test_argument_parsing(self):
        arguments = main_module.parse_arguments(["--text", "--resume"])
        assert arguments.text is True
        assert arguments.resume is True
        assert arguments.tui is False

    def test_text_and_tui_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            main_module.parse_arguments(["--text", "--tui"])

    def test_no_arguments_means_voice(self):
        arguments = main_module.parse_arguments([])
        assert arguments.text is False and arguments.tui is False

    def test_resume_is_passed_through(self, wiring, monkeypatch):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--text", "--resume"])
        captured = {}

        async def text(self):
            captured["resume"] = self.resume

        monkeypatch.setattr(Jarvis, "run_text_mode", text)
        main_module.main()
        assert captured["resume"] is True

    def test_help_prints_usage(self, monkeypatch, capsys):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--help"])
        with pytest.raises(SystemExit) as exit_info:
            main_module.main()
        assert exit_info.value.code == 0
        output = capsys.readouterr().out
        assert "--text" in output and "--tui" in output and "--resume" in output

    def test_default_is_voice_mode(self, wiring, monkeypatch):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py"])
        modes = []

        async def voice(self):
            modes.append("voice")

        async def text(self):
            modes.append("text")

        monkeypatch.setattr(Jarvis, "run_interactive", voice)
        monkeypatch.setattr(Jarvis, "run_text_mode", text)
        main_module.main()
        assert modes == ["voice"]

    def test_text_flag_selects_text_mode(self, wiring, monkeypatch):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--text"])
        modes = []

        async def voice(self):
            modes.append("voice")

        async def text(self):
            modes.append("text")

        monkeypatch.setattr(Jarvis, "run_interactive", voice)
        monkeypatch.setattr(Jarvis, "run_text_mode", text)
        main_module.main()
        assert modes == ["text"]

    def test_tui_flag_selects_voice_mode_with_tui(self, wiring, monkeypatch):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--tui"])
        captured = {}

        async def voice(self):
            captured["tui"] = self.use_tui

        monkeypatch.setattr(Jarvis, "run_interactive", voice)
        monkeypatch.setattr(main_module, "JarvisTUI", lambda: _NullTUI())
        main_module.main()
        assert captured["tui"] is True

    def test_startup_failure_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py"])

        def explode():
            raise RuntimeError("no audio device")

        monkeypatch.setattr(main_module, "AudioHandler", explode)
        with pytest.raises(SystemExit) as exit_info:
            main_module.main()
        assert exit_info.value.code == 1


class RecordingTUI:
    """Captures every call the orchestrator makes into the UI layer."""

    def __init__(self):
        self.events = []
        self.statuses = []
        self.language = None

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append((name, args))

        return record

    def show_welcome(self):
        self.events.append(("show_welcome", ()))

    def update_status(self, status):
        self.statuses.append(status)

    def update_language(self, language):
        self.language = language

    def prompt_confirmation(self, decision):
        self.events.append(("prompt_confirmation", (decision,)))
        return (True, False)


class _NullTUI(RecordingTUI):
    pass


@pytest.fixture
def tui_jarvis(wiring, monkeypatch):
    """A Jarvis running in TUI mode with a recording UI."""
    recording = RecordingTUI()
    monkeypatch.setattr(main_module, "JarvisTUI", lambda: recording)
    instance = Jarvis(use_tui=True)
    instance._tui = recording
    return instance


class TestTuiMode:
    def test_welcome_is_shown_at_startup(self, tui_jarvis):
        assert ("show_welcome", ()) in tui_jarvis._tui.events

    def test_initial_status_and_language_are_published(self, tui_jarvis):
        assert "Ready" in tui_jarvis._tui.statuses
        assert tui_jarvis._tui.language is not None

    def test_confirmation_is_delegated_to_the_tui(self, tui_jarvis):
        assert tui_jarvis._confirm(make_decision()) == (True, False)
        assert any(name == "prompt_confirmation"
                   for name, _ in tui_jarvis._tui.events)

    async def test_a_turn_publishes_both_messages(self, tui_jarvis, wiring):
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
        await tui_jarvis.run_interactive()
        events = dict_events(tui_jarvis._tui.events)
        assert "bonjour" in events["add_user_message"]
        assert "Salut" in events["add_assistant_message"]

    async def test_the_status_walks_through_the_pipeline(self, tui_jarvis, wiring):
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
        await tui_jarvis.run_interactive()
        for status in ("Listening...", "Transcribing...", "Thinking...", "Speaking..."):
            assert status in tui_jarvis._tui.statuses

    async def test_tool_calls_are_logged_to_the_actions_panel(self, tui_jarvis, wiring):
        call = {"function": {"name": "web__fetch", "arguments": {"url": "u"}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        await tui_jarvis.process_user_input("lis le site")
        logged = [args for name, args in tui_jarvis._tui.events if name == "add_action"]
        assert any("web__fetch" in args[0] for args in logged)

    async def test_a_failed_tool_is_logged_as_an_error(self, tui_jarvis, wiring, monkeypatch):
        call = {"function": {"name": "web__fetch", "arguments": {}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        async def failing_call(name, args):
            return ToolResult.error("boom")

        monkeypatch.setattr(tui_jarvis.registry, "call", failing_call)
        await tui_jarvis.process_user_input("lis le site")
        statuses = [args[-1] for name, args in tui_jarvis._tui.events
                    if name == "add_action"]
        assert "error" in statuses

    async def test_language_switch_updates_the_header(self, tui_jarvis, wiring):
        wiring["stt"].script = ["switch to french", "exit"]
        await tui_jarvis.run_interactive()
        assert tui_jarvis._tui.language == "fr"

    async def test_a_recording_error_is_surfaced_and_survived(
        self, tui_jarvis, wiring, monkeypatch
    ):
        buffers = iter([OSError("device lost"), np.zeros(16000, dtype=np.int16)])

        def flaky(**kwargs):
            value = next(buffers)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(wiring["audio"], "record_until_silence", flaky)
        await tui_jarvis.run_interactive()
        messages = dict_events(tui_jarvis._tui.events).get("add_system_message", [])
        assert any("device lost" in str(message) for message in messages)

    async def test_the_display_is_stopped_on_exit(self, tui_jarvis, wiring):
        wiring["stt"].script = ["exit"]
        await tui_jarvis.run_interactive()
        assert any(name == "stop" for name, _ in tui_jarvis._tui.events)


def dict_events(events):
    """Group recorded UI events by method name -> list of first arguments."""
    grouped = {}
    for name, args in events:
        grouped.setdefault(name, []).append(args[0] if args else None)
    return grouped


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

async def test_a_sentence_containing_stop_is_not_an_exit_command(wiring):
    """End to end: a request that merely contains "stop" reaches the LLM."""
    wiring["stt"].script = ["stop the music please", "exit"]
    instance = Jarvis(use_tui=False)
    await instance.run_interactive()
    assert wiring["llm"].chats, "the request should have reached the LLM"


async def test_text_mode_supports_language_switching(wiring, monkeypatch):
    """The voice loop had this inline, so a text session could not switch at all."""
    answers = iter(["switch to french", "exit"])
    monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
    await Jarvis(use_tui=False).run_text_mode()
    assert wiring["stt"].language == "fr"


def test_text_mode_skips_audio_model_loading(wiring, monkeypatch):
    """``--text`` is documented as 'testing without audio I/O'."""
    async def noop(self):
        return None

    monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--text"])
    monkeypatch.setattr(Jarvis, "run_text_mode", noop)
    main_module.main()
    assert wiring["stt"].initialized is False


def test_unknown_flag_is_reported(wiring, monkeypatch, capsys):
    """"--txt" used to start a voice session with the microphone open."""
    async def noop(self):
        return None

    monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--txt"])
    monkeypatch.setattr(Jarvis, "run_interactive", noop)
    monkeypatch.setattr(Jarvis, "run_text_mode", noop)
    with pytest.raises(SystemExit):
        main_module.main()



async def test_playback_uses_the_synthesiser_sample_rate(wiring):
    """A ``*-low`` Piper voice outputs 16 kHz; played at an assumed 22050 it
    would come out chipmunked."""
    wiring["stt"].script = ["bonjour", "exit"]
    wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
    await Jarvis(use_tui=False).run_interactive()
    assert wiring["audio"].played
    assert all(rate == wiring["tts"].sample_rate
               for _, rate in wiring["audio"].played)
