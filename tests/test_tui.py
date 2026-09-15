"""Tests for the Rich terminal interface (tui.py)."""

import io

import pytest
from rich.console import Console

from policy.engine import Decision, Surface
from tools.schema import Risk
from tui import JarvisTUI


def make_decision(surface=Surface.VOICE, risk=Risk.WRITE):
    return Decision(
        tool="fs__write", risk=risk, scope="/tmp/docs",
        surface=surface, summary="fs__write (path=/tmp/docs/a.txt)",
        reason="write access",
    )


@pytest.fixture
def tui():
    instance = JarvisTUI()
    instance.console = Console(file=io.StringIO(), width=120, force_terminal=False)
    return instance


def render(renderable, width=120) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=width, force_terminal=False).print(renderable)
    return buffer.getvalue()


class TestState:
    def test_starts_empty(self, tui):
        assert list(tui.chat_history) == []
        assert list(tui.actions_log) == []

    def test_add_user_message(self, tui):
        tui.add_user_message("bonjour")
        assert tui.chat_history[0]["role"] == "user"
        assert tui.chat_history[0]["content"] == "bonjour"
        assert tui.chat_history[0]["timestamp"]

    def test_add_assistant_message(self, tui):
        tui.add_assistant_message("salut")
        assert tui.chat_history[0]["role"] == "assistant"

    def test_add_system_message(self, tui):
        tui.add_system_message("started")
        assert tui.chat_history[0]["role"] == "system"

    def test_messages_keep_their_order(self, tui):
        tui.add_user_message("one")
        tui.add_assistant_message("two")
        assert [m["content"] for m in tui.chat_history] == ["one", "two"]

    def test_add_action_defaults_to_info(self, tui):
        tui.add_action("fetch_web_page", "https://example.com")
        assert tui.actions_log[0]["status"] == "info"

    def test_add_action_records_status(self, tui):
        tui.add_action("fetch_web_page", "done", "success")
        assert tui.actions_log[0]["status"] == "success"

    def test_clear_history(self, tui):
        tui.add_user_message("one")
        tui.clear_history()
        assert list(tui.chat_history) == []

    def test_clear_actions(self, tui):
        tui.add_action("a", "b")
        tui.clear_actions()
        assert list(tui.actions_log) == []

    def test_update_status(self, tui):
        tui.update_status("Listening...")
        assert tui.current_status == "Listening..."

    def test_update_language(self, tui):
        tui.update_language("fr")
        assert tui.current_language == "fr"

    def test_refresh_without_a_live_display_is_safe(self, tui):
        tui.update_status("Ready")  # no Live started - must not raise


class TestRendering:
    def test_header_shows_status_and_language(self, tui):
        tui.current_status = "Listening"
        tui.current_language = "fr"
        output = render(tui._make_header())
        assert "JARVIS" in output
        assert "Listening" in output
        assert "FR" in output

    def test_empty_chat_panel_has_a_placeholder(self, tui):
        assert "No conversation yet" in render(tui._make_chat_panel())

    def test_chat_panel_shows_messages(self, tui):
        tui.add_user_message("bonjour")
        tui.add_assistant_message("salut")
        output = render(tui._make_chat_panel())
        assert "bonjour" in output and "salut" in output

    def test_chat_panel_drops_the_oldest_messages(self, tui):
        for index in range(15):
            tui.add_user_message(f"message{index:02d}")
        output = render(tui._make_chat_panel())
        assert "message00" not in output
        assert "message04" not in output

    def test_empty_actions_panel_has_a_placeholder(self, tui):
        assert "No actions performed yet" in render(tui._make_actions_panel())

    def test_actions_panel_shows_entries(self, tui):
        tui.add_action("fetch_web_page", "example.com", "success")
        output = render(tui._make_actions_panel())
        assert "fetch_web_page" in output

    def test_actions_panel_truncates_long_details(self, tui):
        tui.add_action("op", "d" * 200)
        assert "..." in render(tui._make_actions_panel())

    def test_help_panel_lists_the_commands(self, tui):
        output = render(tui._make_help_panel())
        assert "exit" in output and "Ctrl+C" in output

    def test_layout_has_every_region(self, tui):
        layout = tui._make_layout()
        tui._update_layout(layout)
        for name in ("header", "chat", "actions", "help"):
            assert layout[name] is not None

    def test_welcome_screen_renders(self, tui):
        tui.show_welcome()
        assert "JARVIS" in tui.console.file.getvalue()

    def test_unicode_content_renders(self, tui):
        tui.add_user_message("crée un fichier café.txt 🎉")
        assert "café" in render(tui._make_chat_panel())


class TestConfirmationPrompt:
    def _answer(self, tui, answers, decision=None):
        responses = iter(answers)
        tui.console.input = lambda *args, **kwargs: next(responses)
        return tui.prompt_confirmation(decision or make_decision())

    def test_yes(self, tui):
        assert self._answer(tui, ["y"]) == (True, False)

    def test_always(self, tui):
        assert self._answer(tui, ["a"]) == (True, True)

    def test_no(self, tui):
        assert self._answer(tui, ["n"]) == (False, False)

    def test_case_and_whitespace_are_tolerated(self, tui):
        assert self._answer(tui, [" Y "]) == (True, False)

    def test_invalid_answers_reprompt(self, tui):
        assert self._answer(tui, ["maybe", "", "n"]) == (False, False)

    def test_the_real_arguments_are_shown(self, tui):
        """"fs__write (path=...)" is a decision someone can make."""
        self._answer(tui, ["n"])
        output = tui.console.file.getvalue()
        assert "path=/tmp/docs/a.txt" in output
        assert "write access" in output

    def test_the_risk_level_is_shown(self, tui):
        self._answer(tui, ["n"])
        assert "WRITE" in tui.console.file.getvalue()

    def test_the_scope_of_always_is_shown(self, tui):
        self._answer(tui, ["n"])
        assert "/tmp/docs" in tui.console.file.getvalue()

    def test_a_terminal_decision_does_not_offer_always(self, tui):
        self._answer(tui, ["n"], make_decision(Surface.TERMINAL, Risk.DESTRUCTIVE))
        output = tui.console.file.getvalue()
        assert "stop asking" not in output
        assert "keyboard" in output

    def test_always_is_refused_on_a_terminal_decision(self, tui):
        result = self._answer(tui, ["a", "y"], make_decision(Surface.TERMINAL))
        assert result == (True, False)

    def test_live_display_is_suspended_and_restored(self, tui):
        events = []

        class FakeLive:
            def start(self_inner):
                events.append("start")

            def stop(self_inner):
                events.append("stop")

        tui.live = FakeLive()
        self._answer(tui, ["y"])
        assert events == ["stop", "start"]


class TestLifecycle:
    def test_stop_without_start_is_safe(self, tui):
        tui.stop()

    def test_start_then_stop(self, tui, monkeypatch):
        started = []

        class FakeLive:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def start(self_inner):
                started.append("start")

            def stop(self_inner):
                started.append("stop")

            def update(self_inner, layout):
                started.append("update")

        monkeypatch.setattr("tui.Live", FakeLive)
        tui.start()
        tui.refresh()
        tui.stop()
        assert started == ["start", "update", "stop"]


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

def test_history_is_capped_in_memory(tui):
    """A long session would otherwise hold the whole transcript, and only the
    last handful is ever rendered. The record lives in SQLite now."""
    for _ in range(5000):
        tui.add_user_message("x" * 200)
    assert len(tui.chat_history) <= 1000


def test_the_actions_log_is_capped_too(tui):
    for index in range(5000):
        tui.add_action("op", str(index))
    assert len(tui.actions_log) <= 1000


class TestPanelFitting:
    """The newest message is the one the user is waiting for."""

    def test_the_latest_message_is_always_shown(self, tui):
        for index in range(15):
            tui.add_user_message(f"message{index:02d}")
        assert "message14" in render(tui._make_chat_panel())

    def test_the_latest_message_is_shown_on_a_short_terminal(self, tui):
        tui.console = Console(file=io.StringIO(), width=100, height=14)
        for index in range(15):
            tui.add_user_message(f"message{index:02d}")
        assert "message14" in render(tui._make_chat_panel(), width=100)

    def test_a_taller_terminal_shows_more(self, tui):
        for index in range(30):
            tui.add_user_message(f"message{index:02d}")

        tui.console = Console(file=io.StringIO(), width=100, height=16)
        short = len(tui._fitting_messages(80, tui._body_height()))
        tui.console = Console(file=io.StringIO(), width=100, height=60)
        tall = len(tui._fitting_messages(80, tui._body_height()))
        assert tall > short

    def test_long_messages_take_more_room(self, tui):
        tui.console = Console(file=io.StringIO(), width=100, height=30)
        for _ in range(20):
            tui.add_user_message("court")
        short_fit = len(tui._fitting_messages(80, tui._body_height()))

        tui.clear_history()
        for _ in range(20):
            tui.add_user_message("x" * 400)
        long_fit = len(tui._fitting_messages(80, tui._body_height()))
        assert long_fit < short_fit

    def test_one_message_always_fits(self, tui):
        """Even a message too long for the panel is better than an empty one."""
        tui.console = Console(file=io.StringIO(), width=60, height=10)
        tui.add_user_message("x" * 2000)
        assert len(tui._fitting_messages(40, tui._body_height())) == 1

    def test_panels_fit_a_short_terminal(self, tui):
        tui.console = Console(file=io.StringIO(), width=80, height=24)
        output = render(tui._make_chat_panel(), width=80)
        assert len(output.splitlines()) <= tui._body_height()
