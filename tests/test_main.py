"""Tests for the orchestration loop (main.py)."""

import builtins

import numpy as np
import pytest

import main as main_module
from main import Jarvis


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

    def __init__(self, script=None):
        self.script = list(script or [])
        self.chats = []          # user messages passed to chat()
        self.follow_ups = 0      # continue_after_tools() calls
        self.tool_results = []

    def _next(self):
        if self.script:
            return self.script.pop(0)
        return {"response": "ok", "tool_calls": None}

    def chat(self, message):
        self.chats.append(message)
        return self._next()

    def continue_after_tools(self):
        self.follow_ups += 1
        return self._next()

    def add_tool_result(self, name, result):
        self.tool_results.append((name, result))


class FakeExecutor:
    def __init__(self, confirmation_callback=None):
        self.confirmation_callback = confirmation_callback
        self.executed = []

    def execute_tool_call(self, tool_call):
        self.executed.append(tool_call)
        return "tool ok"


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


@pytest.fixture
def wiring(monkeypatch):
    """Replace every subsystem with an inspectable fake."""
    parts = {
        "audio": FakeAudio(),
        "stt": FakeSTT(),
        "llm": FakeLLM(),
        "tts": FakeTTS(),
    }
    monkeypatch.setattr(main_module, "AudioHandler", lambda: parts["audio"])
    monkeypatch.setattr(main_module, "STTModule", lambda: parts["stt"])
    monkeypatch.setattr(main_module, "LLMModule", lambda: parts["llm"])
    monkeypatch.setattr(main_module, "TTSModule", lambda: parts["tts"])
    monkeypatch.setattr(main_module, "ActionExecutor", FakeExecutor)
    return parts


@pytest.fixture
def jarvis(wiring):
    return Jarvis(use_tui=False)


class TestStartup:
    def test_loads_both_models(self, jarvis, wiring):
        assert wiring["stt"].initialized is True
        assert wiring["tts"].initialized is True

    def test_executor_gets_the_confirmation_callback(self, jarvis):
        assert jarvis.executor.confirmation_callback == jarvis._confirmation_callback

    def test_no_tui_by_default(self, jarvis):
        assert jarvis.use_tui is False
        assert jarvis.tui is None


class TestConfirmationPrompt:
    @pytest.mark.parametrize("answer,expected", [
        ("y", (True, False)),
        ("a", (True, True)),
        ("n", (False, False)),
        ("Y", (True, False)),
        (" a ", (True, True)),
    ])
    def test_answers(self, jarvis, monkeypatch, answer, expected):
        monkeypatch.setattr(builtins, "input", lambda *a: answer)
        assert jarvis._confirmation_callback("delete x", "delete_file:/tmp") == expected

    def test_invalid_answer_reprompts(self, jarvis, monkeypatch):
        answers = iter(["maybe", "", "y"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        assert jarvis._confirmation_callback("delete x", "item") == (True, False)


class TestProcessUserInput:
    def test_plain_response_is_returned(self, jarvis, wiring):
        wiring["llm"].script = [{"response": "Bonjour", "tool_calls": None}]
        assert jarvis.process_user_input("salut") == "Bonjour"

    def test_no_tools_means_a_single_llm_round(self, jarvis, wiring):
        wiring["llm"].script = [{"response": "Bonjour", "tool_calls": None}]
        jarvis.process_user_input("salut")
        assert len(wiring["llm"].chats) == 1

    def test_tool_calls_are_executed(self, jarvis, wiring):
        call = {"function": {"name": "fetch_web_page", "arguments": {"url": "u"}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "Voici le résumé", "tool_calls": None},
        ]
        assert jarvis.process_user_input("lis le site") == "Voici le résumé"
        assert jarvis.executor.executed == [call]

    def test_tool_results_are_fed_back(self, jarvis, wiring):
        call = {"function": {"name": "fetch_web_page", "arguments": {"url": "u"}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        jarvis.process_user_input("lis le site")
        assert wiring["llm"].tool_results == [("fetch_web_page", "tool ok")]

    def test_a_second_round_of_tool_calls_also_runs(self, jarvis, wiring):
        """The follow-up turn may ask for more tools; they must be executed."""
        first = {"function": {"name": "execute_file_operation", "arguments": {}}}
        second = {"function": {"name": "fetch_web_page", "arguments": {}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [first]},
            {"response": "", "tool_calls": [second]},
            {"response": "fini", "tool_calls": None},
        ]
        assert jarvis.process_user_input("fais les deux") == "fini"
        assert jarvis.executor.executed == [first, second]

    def test_the_tool_loop_is_bounded(self, jarvis, wiring):
        """A model that keeps asking for tools must not loop forever."""
        call = {"function": {"name": "fetch_web_page", "arguments": {}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]} for _ in range(50)
        ]
        jarvis.process_user_input("boucle")
        assert len(jarvis.executor.executed) == FakeLLM.MAX_TOOL_ITERATIONS

    def test_the_follow_up_is_not_a_user_turn(self, jarvis, wiring):
        call = {"function": {"name": "fetch_web_page", "arguments": {}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        jarvis.process_user_input("lis le site")
        assert wiring["llm"].chats == ["lis le site"]
        assert wiring["llm"].follow_ups == 1

    def test_several_tool_calls_all_run(self, jarvis, wiring):
        calls = [
            {"function": {"name": "execute_file_operation", "arguments": {}}},
            {"function": {"name": "fetch_web_page", "arguments": {}}},
        ]
        wiring["llm"].script = [
            {"response": "", "tool_calls": calls},
            {"response": "done", "tool_calls": None},
        ]
        jarvis.process_user_input("fais deux choses")
        assert len(jarvis.executor.executed) == 2


class TestVoiceLoop:
    def test_exit_command_ends_the_loop(self, jarvis, wiring):
        wiring["stt"].script = ["exit"]
        jarvis.run_interactive()
        assert wiring["tts"].spoken[-1] in ("Goodbye!", "Au revoir!")

    @pytest.mark.parametrize("phrase", ["exit", "quit", "goodbye", "au revoir"])
    def test_exit_synonyms(self, wiring, phrase):
        wiring["stt"].script = [phrase]
        Jarvis(use_tui=False).run_interactive()
        assert wiring["tts"].spoken

    def test_a_normal_turn_is_spoken_then_the_loop_continues(self, jarvis, wiring):
        wiring["stt"].script = ["quelle heure est-il", "exit"]
        wiring["llm"].script = [{"response": "Il est midi", "tool_calls": None}]
        jarvis.run_interactive()
        assert "Il est midi" in wiring["tts"].spoken

    def test_switch_to_french(self, jarvis, wiring):
        wiring["stt"].script = ["switch to french", "exit"]
        jarvis.run_interactive()
        assert wiring["stt"].language == "fr"

    def test_switch_to_english(self, jarvis, wiring):
        wiring["stt"].script = ["parle anglais", "exit"]
        jarvis.run_interactive()
        assert wiring["stt"].language == "en"

    def test_language_switch_does_not_reach_the_llm(self, jarvis, wiring):
        wiring["stt"].script = ["en français", "exit"]
        jarvis.run_interactive()
        assert wiring["llm"].chats == []

    def test_empty_transcription_is_skipped(self, jarvis, wiring):
        wiring["stt"].script = ["", "exit"]
        jarvis.run_interactive()
        assert wiring["llm"].chats == []

    def test_short_recording_is_skipped(self, jarvis, wiring, monkeypatch):
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
        jarvis.run_interactive()
        assert len(transcribed) == 1

    def test_transcription_error_does_not_kill_the_loop(self, jarvis, wiring, monkeypatch):
        results = iter([RuntimeError("stt down"), "exit"])

        def flaky(*args, **kwargs):
            value = next(results)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(wiring["stt"], "transcribe", flaky)
        jarvis.run_interactive()
        assert wiring["tts"].spoken

    def test_empty_synthesis_falls_back_to_text(self, jarvis, wiring, monkeypatch, capsys):
        """Nothing to play means say it in the terminal, not stay silent."""
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
        monkeypatch.setattr(
            wiring["tts"], "synthesize",
            lambda text: (np.array([], dtype=np.int16), 16000),
        )
        jarvis.run_interactive()
        assert "Salut" in capsys.readouterr().out

    def test_tts_failure_falls_back_to_text(self, jarvis, wiring, monkeypatch, capsys):
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]

        def broken(text):
            raise RuntimeError("piper missing")

        monkeypatch.setattr(wiring["tts"], "synthesize", broken)
        jarvis.run_interactive()
        assert "Salut" in capsys.readouterr().out

    def test_processing_error_is_reported_not_raised(self, jarvis, wiring, monkeypatch):
        wiring["stt"].script = ["bonjour", "exit"]

        def broken(text):
            raise RuntimeError("llm down")

        monkeypatch.setattr(jarvis, "process_user_input", broken)
        jarvis.run_interactive()
        assert any("error" in text.lower() for text in wiring["tts"].spoken)


class TestTextMode:
    def test_processes_input_and_exits(self, jarvis, wiring, monkeypatch, capsys):
        answers = iter(["bonjour", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
        jarvis.run_text_mode()
        assert "Salut" in capsys.readouterr().out

    def test_blank_lines_are_ignored(self, jarvis, wiring, monkeypatch):
        answers = iter(["", "   ", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        jarvis.run_text_mode()
        assert wiring["llm"].chats == []

    def test_ctrl_c_exits_cleanly(self, jarvis, wiring, monkeypatch):
        def interrupt(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr(builtins, "input", interrupt)
        jarvis.run_text_mode()

    def test_errors_are_reported_not_raised(self, jarvis, wiring, monkeypatch, capsys):
        answers = iter(["bonjour", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))

        def broken(text):
            raise RuntimeError("llm down")

        monkeypatch.setattr(jarvis, "process_user_input", broken)
        jarvis.run_text_mode()
        assert "error" in capsys.readouterr().out.lower()

    def test_no_audio_is_played(self, jarvis, wiring, monkeypatch):
        answers = iter(["bonjour", "exit"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        jarvis.run_text_mode()
        assert wiring["audio"].played == []


class TestEntryPoint:
    @pytest.fixture(autouse=True)
    def isolated_cwd(self, tmp_path, monkeypatch):
        """``main()`` configures a file logger; keep jarvis.log out of the repo."""
        monkeypatch.chdir(tmp_path)

    def test_help_prints_usage(self, monkeypatch, capsys):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--help"])
        main_module.main()
        output = capsys.readouterr().out
        assert "--text" in output and "--tui" in output

    def test_default_is_voice_mode(self, wiring, monkeypatch):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py"])
        modes = []
        monkeypatch.setattr(Jarvis, "run_interactive", lambda self: modes.append("voice"))
        monkeypatch.setattr(Jarvis, "run_text_mode", lambda self: modes.append("text"))
        main_module.main()
        assert modes == ["voice"]

    def test_text_flag_selects_text_mode(self, wiring, monkeypatch):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--text"])
        modes = []
        monkeypatch.setattr(Jarvis, "run_interactive", lambda self: modes.append("voice"))
        monkeypatch.setattr(Jarvis, "run_text_mode", lambda self: modes.append("text"))
        main_module.main()
        assert modes == ["text"]

    def test_tui_flag_selects_voice_mode_with_tui(self, wiring, monkeypatch):
        monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--tui"])
        captured = {}
        monkeypatch.setattr(
            Jarvis, "run_interactive", lambda self: captured.update(tui=self.use_tui)
        )
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

    def prompt_confirmation(self, description, item):
        self.events.append(("prompt_confirmation", (description, item)))
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
        assert tui_jarvis._confirmation_callback("delete x", "item") == (True, False)
        assert any(name == "prompt_confirmation"
                   for name, _ in tui_jarvis._tui.events)

    def test_a_turn_publishes_both_messages(self, tui_jarvis, wiring):
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
        tui_jarvis.run_interactive()
        events = dict_events(tui_jarvis._tui.events)
        assert "bonjour" in events["add_user_message"]
        assert "Salut" in events["add_assistant_message"]

    def test_the_status_walks_through_the_pipeline(self, tui_jarvis, wiring):
        wiring["stt"].script = ["bonjour", "exit"]
        wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
        tui_jarvis.run_interactive()
        for status in ("Listening...", "Transcribing...", "Thinking...", "Speaking..."):
            assert status in tui_jarvis._tui.statuses

    def test_tool_calls_are_logged_to_the_actions_panel(self, tui_jarvis, wiring):
        call = {"function": {"name": "fetch_web_page", "arguments": {"url": "u"}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        tui_jarvis.process_user_input("lis le site")
        logged = [args for name, args in tui_jarvis._tui.events if name == "add_action"]
        assert any("fetch_web_page" in args[0] for args in logged)

    def test_a_failed_tool_is_logged_as_an_error(self, tui_jarvis, wiring, monkeypatch):
        call = {"function": {"name": "fetch_web_page", "arguments": {}}}
        wiring["llm"].script = [
            {"response": "", "tool_calls": [call]},
            {"response": "done", "tool_calls": None},
        ]
        monkeypatch.setattr(
            tui_jarvis.executor, "execute_tool_call", lambda c: "Error: boom"
        )
        tui_jarvis.process_user_input("lis le site")
        statuses = [args[-1] for name, args in tui_jarvis._tui.events
                    if name == "add_action"]
        assert "error" in statuses

    def test_language_switch_updates_the_header(self, tui_jarvis, wiring):
        wiring["stt"].script = ["switch to french", "exit"]
        tui_jarvis.run_interactive()
        assert tui_jarvis._tui.language == "fr"

    def test_a_recording_error_is_surfaced_and_survived(
        self, tui_jarvis, wiring, monkeypatch
    ):
        buffers = iter([OSError("device lost"), np.zeros(16000, dtype=np.int16)])

        def flaky(**kwargs):
            value = next(buffers)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(wiring["audio"], "record_until_silence", flaky)
        tui_jarvis.run_interactive()
        messages = dict_events(tui_jarvis._tui.events).get("add_system_message", [])
        assert any("device lost" in str(message) for message in messages)

    def test_the_display_is_stopped_on_exit(self, tui_jarvis, wiring):
        wiring["stt"].script = ["exit"]
        tui_jarvis.run_interactive()
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

@pytest.mark.xfail(strict=True, reason="BUG-22: exit words are matched as substrings")
def test_a_sentence_containing_stop_is_not_an_exit_command(wiring):
    """'stop' matches anywhere in the transcript, so "arrête la musique",
    "stop the timer" or any sentence containing the word quits Jarvis."""
    wiring["stt"].script = ["stop the music please", "exit"]
    instance = Jarvis(use_tui=False)
    instance.run_interactive()
    assert wiring["llm"].chats, "the request should have reached the LLM"


@pytest.mark.xfail(strict=True, reason="BUG-23: text mode cannot switch language")
def test_text_mode_supports_language_switching(wiring, monkeypatch):
    answers = iter(["switch to french", "exit"])
    monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
    Jarvis(use_tui=False).run_text_mode()
    assert wiring["stt"].language == "fr"


@pytest.mark.xfail(strict=True, reason="BUG-24: text mode still loads the audio models")
def test_text_mode_skips_audio_model_loading(wiring, monkeypatch):
    """``--text`` is documented as 'testing without audio I/O', yet it still
    downloads and loads Whisper and probes Piper before the first prompt."""
    monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--text"])
    monkeypatch.setattr(Jarvis, "run_text_mode", lambda self: None)
    main_module.main()
    assert wiring["stt"].initialized is False


@pytest.mark.xfail(strict=True, reason="BUG-25: unknown CLI flags are silently ignored")
def test_unknown_flag_is_reported(wiring, monkeypatch, capsys):
    monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--txt"])
    monkeypatch.setattr(Jarvis, "run_interactive", lambda self: None)
    monkeypatch.setattr(Jarvis, "run_text_mode", lambda self: None)
    with pytest.raises(SystemExit):
        main_module.main()



def test_playback_uses_the_synthesiser_sample_rate(wiring):
    """A ``*-low`` Piper voice outputs 16 kHz; played at an assumed 22050 it
    would come out chipmunked."""
    wiring["stt"].script = ["bonjour", "exit"]
    wiring["llm"].script = [{"response": "Salut", "tool_calls": None}]
    Jarvis(use_tui=False).run_interactive()
    assert wiring["audio"].played
    assert all(rate == wiring["tts"].sample_rate
               for _, rate in wiring["audio"].played)
