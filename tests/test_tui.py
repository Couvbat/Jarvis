"""Tests for the Rich terminal interface (tui.py)."""

import io

import pytest
from rich.console import Console

from tui import JarvisTUI


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
        assert tui.chat_history == []
        assert tui.actions_log == []

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
        assert tui.chat_history == []

    def test_clear_actions(self, tui):
        tui.add_action("a", "b")
        tui.clear_actions()
        assert tui.actions_log == []

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
    def _answer(self, tui, answers):
        responses = iter(answers)
        tui.console.input = lambda *args, **kwargs: next(responses)
        return tui.prompt_confirmation("delete /tmp/x", "delete_file:/tmp")

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

    def test_the_action_is_shown_to_the_user(self, tui):
        self._answer(tui, ["n"])
        assert "delete /tmp/x" in tui.console.file.getvalue()

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

@pytest.mark.xfail(strict=True, reason="BUG-28: TUI panels grow without bound")
def test_history_is_capped_in_memory(tui):
    """Only the last 10 messages are *rendered*, but every message is kept.

    A long-running session accumulates the whole transcript in RAM with no
    trimming and no way to persist or page through it.
    """
    for _ in range(5000):
        tui.add_user_message("x" * 200)
    assert len(tui.chat_history) <= 1000


@pytest.mark.xfail(
    strict=True, reason="BUG-29: the newest messages are clipped out of the chat panel"
)
def test_chat_panel_always_shows_the_latest_message(tui):
    """The panel selects the last 10 messages but is pinned to ``height=20``.

    Each message renders as two lines (text + spacer) plus padding, so only
    eight fit: the two most recent turns - usually Jarvis's actual answer -
    are cut off the bottom of the panel.
    """
    for index in range(15):
        tui.add_user_message(f"message{index:02d}")
    assert "message14" in render(tui._make_chat_panel())


@pytest.mark.xfail(strict=True, reason="BUG-30: panel heights do not adapt")
def test_panels_adapt_to_small_terminals(tui):
    """``height=20`` on the chat panel and ``height=12`` on the actions panel
    overflow a terminal shorter than ~40 rows."""
    output = render(tui._make_chat_panel(), width=80)
    assert len(output.splitlines()) < 20
