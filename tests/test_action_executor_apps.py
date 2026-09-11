"""Application launching in ActionExecutor.launch_application."""

import subprocess

import pytest


class FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def run_calls(monkeypatch):
    """Capture ``subprocess.run`` invocations from action_executor."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return FakeCompleted(stdout="fake stdout")

    monkeypatch.setattr("action_executor.subprocess.run", fake_run)
    return calls


@pytest.fixture
def popen_calls(monkeypatch):
    """Capture ``subprocess.Popen`` invocations from action_executor."""
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            calls.append({"cmd": cmd, **kwargs})

    monkeypatch.setattr("action_executor.subprocess.Popen", FakePopen)
    return calls


class TestCommandWhitelist:
    def test_unlisted_command_is_refused(self, executor, run_calls):
        result = executor.launch_application("curl")
        assert "not in whitelist" in result
        assert run_calls == []

    def test_unlisted_command_is_refused_before_confirmation(
        self, executor, approve_all, run_calls
    ):
        executor.launch_application("curl")
        assert approve_all.calls == []

    def test_listed_command_runs(self, executor, run_calls):
        result = executor.launch_application("ls")
        assert "fake stdout" in result
        assert run_calls[0]["cmd"] == ["ls"]

    def test_arguments_are_passed_as_a_list(self, executor, run_calls):
        executor.launch_application("ls", ["-la", "/tmp"])
        assert run_calls[0]["cmd"] == ["ls", "-la", "/tmp"]

    def test_no_shell_is_used(self, executor, run_calls):
        executor.launch_application("ls", ["-la"])
        assert run_calls[0].get("shell") in (None, False)

    def test_shell_metacharacters_are_not_interpreted(self, executor, run_calls):
        executor.launch_application("echo", ["hi; rm -rf /"])
        assert run_calls[0]["cmd"] == ["echo", "hi; rm -rf /"]
        assert run_calls[0].get("shell") in (None, False)

    def test_cli_commands_have_a_timeout(self, executor, run_calls):
        executor.launch_application("ls")
        assert run_calls[0].get("timeout") is not None


class TestGuiApplications:
    @pytest.mark.parametrize("app", ["code", "firefox", "nautilus"])
    def test_gui_apps_are_detached(self, executor, app, popen_calls, run_calls):
        result = executor.launch_application(app)
        assert f"Launched {app}" in result
        assert popen_calls[0]["cmd"] == [app]
        assert popen_calls[0]["start_new_session"] is True
        assert run_calls == []

    def test_gui_output_is_discarded(self, executor, popen_calls):
        executor.launch_application("firefox")
        assert popen_calls[0]["stdout"] is subprocess.DEVNULL
        assert popen_calls[0]["stderr"] is subprocess.DEVNULL

    def test_gui_args_are_forwarded(self, executor, popen_calls):
        executor.launch_application("firefox", ["https://example.com"])
        assert popen_calls[0]["cmd"] == ["firefox", "https://example.com"]


class TestOutputHandling:
    def test_stderr_is_included(self, executor, monkeypatch):
        monkeypatch.setattr(
            "action_executor.subprocess.run",
            lambda cmd, **kw: FakeCompleted(stdout="out", stderr="bad"),
        )
        result = executor.launch_application("ls")
        assert "out" in result and "bad" in result

    def test_empty_output_reports_execution(self, executor, monkeypatch):
        monkeypatch.setattr(
            "action_executor.subprocess.run", lambda cmd, **kw: FakeCompleted()
        )
        assert "Command executed" in executor.launch_application("ls")

    def test_long_output_is_truncated(self, executor, monkeypatch):
        monkeypatch.setattr(
            "action_executor.subprocess.run",
            lambda cmd, **kw: FakeCompleted(stdout="x" * 5000),
        )
        result = executor.launch_application("ls")
        assert "(truncated)" in result
        assert result.count("x") == 1000

    def test_timeout_is_reported_not_raised(self, executor, monkeypatch):
        def raise_timeout(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 5)

        monkeypatch.setattr("action_executor.subprocess.run", raise_timeout)
        assert "timed out" in executor.launch_application("ls")

    def test_missing_binary_is_reported_not_raised(self, executor, monkeypatch):
        def raise_not_found(cmd, **kwargs):
            raise FileNotFoundError(cmd)

        monkeypatch.setattr("action_executor.subprocess.run", raise_not_found)
        assert "Error launching application" in executor.launch_application("ls")


class TestConfirmation:
    def test_requires_confirmation(self, executor, approve_all, run_calls):
        executor.launch_application("ls")
        assert approve_all.calls[0][1] == "ls"

    def test_confirmation_item_includes_arguments(
        self, executor, approve_all, run_calls
    ):
        executor.launch_application("ls", ["-la"])
        assert approve_all.calls[0][1] == "ls -la"

    def test_denied_confirmation_runs_nothing(self, executor, deny_all, run_calls):
        executor.confirmation_callback = deny_all
        result = executor.launch_application("ls")
        assert "cancelled by user" in result
        assert run_calls == []

    def test_whitelisting_covers_the_exact_command_only(
        self, executor, approve_and_whitelist, run_calls
    ):
        executor.confirmation_callback = approve_and_whitelist
        executor.launch_application("ls", ["-la"])
        executor.launch_application("ls", ["-la"])
        executor.launch_application("ls", ["-l"])
        assert len(approve_and_whitelist.calls) == 2


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason="BUG-02: a command string with args is not split")
def test_command_string_containing_arguments_should_run(executor, run_calls):
    """The whitelist check splits on spaces but the exec does not.

    ``launch_application("ls -la")`` passes the whitelist as ``ls`` and is then
    exec'd as a single binary literally named ``"ls -la"``.
    """
    result = executor.launch_application("ls -la")
    assert run_calls and run_calls[0]["cmd"] == ["ls", "-la"]
    assert "Error" not in result


@pytest.mark.xfail(strict=True, reason="BUG-06: absolute paths bypass the whitelist check")
def test_absolute_path_to_a_listed_command_should_be_normalised(executor, run_calls):
    """``/bin/ls`` is not the string ``ls``, so it is refused; ``./rm`` would be
    refused too while a crafted ``rm`` on PATH would not.  The check should
    compare resolved basenames, not raw strings."""
    result = executor.launch_application("/bin/ls")
    assert "not in whitelist" not in result
