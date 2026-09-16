"""Tests for the application launcher and its whitelist (tools/local/apps.py)."""

import subprocess

import pytest

from tools.local import apps as apps_module
from tools.local.apps import AppTools, build_tools
from tools.registry import ToolRegistry
from tools.schema import Risk


class FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def on_path(monkeypatch):
    """Pretend a fixed set of binaries exists on PATH."""
    known = {
        "ls": "/bin/ls",
        "echo": "/bin/echo",
        "firefox": "/usr/bin/firefox",
        "rm": "/bin/rm",
    }
    monkeypatch.setattr(apps_module.shutil, "which", lambda name: known.get(name))
    return known


@pytest.fixture
def run_calls(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return FakeCompleted(stdout="fake stdout")

    monkeypatch.setattr(apps_module.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def popen_calls(monkeypatch):
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            calls.append({"cmd": cmd, **kwargs})

    monkeypatch.setattr(apps_module.subprocess, "Popen", FakePopen)
    return calls


@pytest.fixture
def tools(on_path):
    return AppTools(["ls", "echo", "firefox"], gui_applications=["firefox"])


class TestWhitelist:
    def test_an_unlisted_command_is_refused(self, tools, run_calls):
        result = tools.launch("rm")
        assert result.ok is False
        assert "not in the command whitelist" in result.content
        assert run_calls == []

    def test_a_listed_command_runs(self, tools, run_calls):
        assert tools.launch("ls").ok is True
        assert run_calls[0]["cmd"] == ["/bin/ls"]

    def test_an_absolute_path_to_a_listed_command_is_accepted(self, tools, run_calls):
        """"/bin/ls" and "ls" name the same program; string comparison missed that."""
        assert tools.launch("/bin/ls").ok is True

    def test_a_lookalike_elsewhere_on_disk_is_refused(self, tools, run_calls, tmp_path):
        """A binary with the right basename in the wrong place is not the same
        program, and basename matching alone would wave it through."""
        impostor = tmp_path / "ls"
        impostor.write_text("#!/bin/sh\n")
        result = tools.launch(str(impostor))
        assert result.ok is False
        assert "is not the 'ls' on PATH" in result.content
        assert run_calls == []

    def test_a_relative_command_path_is_refused(self, tools, run_calls):
        result = tools.launch("./ls")
        assert result.ok is False
        assert "relative command path" in result.content

    def test_a_command_missing_from_path_is_reported(self, tools, run_calls):
        instance = AppTools(["ghost"])
        assert "command not found" in instance.launch("ghost").content

    def test_an_empty_whitelist_allows_nothing(self, on_path, run_calls):
        assert AppTools([]).launch("ls").ok is False


class TestArgumentHandling:
    def test_a_command_string_with_arguments_is_split(self, tools, run_calls):
        """The whitelist check split on spaces but the exec did not, so this
        used to be looked up as a binary literally named "ls -la"."""
        result = tools.launch("ls -la")
        assert result.ok is True
        assert run_calls[0]["cmd"] == ["/bin/ls", "-la"]

    def test_explicit_args_are_appended(self, tools, run_calls):
        tools.launch("ls", ["-la", "/tmp"])
        assert run_calls[0]["cmd"] == ["/bin/ls", "-la", "/tmp"]

    def test_inline_and_explicit_args_combine(self, tools, run_calls):
        tools.launch("ls -l", ["/tmp"])
        assert run_calls[0]["cmd"] == ["/bin/ls", "-l", "/tmp"]

    def test_a_single_string_arg_is_accepted(self, tools, run_calls):
        tools.launch("echo", "hello")
        assert run_calls[0]["cmd"] == ["/bin/echo", "hello"]

    def test_quoted_arguments_survive_splitting(self, tools, run_calls):
        tools.launch('echo "hello world"')
        assert run_calls[0]["cmd"] == ["/bin/echo", "hello world"]

    def test_no_shell_is_used(self, tools, run_calls):
        tools.launch("echo", ["hi; rm -rf /"])
        assert run_calls[0]["cmd"] == ["/bin/echo", "hi; rm -rf /"]
        assert run_calls[0].get("shell") in (None, False)

    def test_unbalanced_quotes_are_reported(self, tools, run_calls):
        result = tools.launch('echo "unclosed')
        assert result.ok is False
        assert "could not parse" in result.content

    def test_an_empty_command_is_reported(self, tools):
        assert tools.launch("").ok is False
        assert tools.launch("   ").ok is False


class TestExecution:
    def test_output_is_returned(self, tools, run_calls):
        assert "fake stdout" in tools.launch("ls").content

    def test_stderr_is_included(self, tools, monkeypatch):
        monkeypatch.setattr(
            apps_module.subprocess, "run",
            lambda cmd, **kw: FakeCompleted(stdout="out", stderr="warn"),
        )
        content = tools.launch("ls").content
        assert "out" in content and "warn" in content

    def test_a_nonzero_exit_is_an_error(self, tools, monkeypatch):
        monkeypatch.setattr(
            apps_module.subprocess, "run",
            lambda cmd, **kw: FakeCompleted(stdout="boom", returncode=2),
        )
        result = tools.launch("ls")
        assert result.ok is False
        assert "status 2" in result.content

    def test_empty_output_still_reports_success(self, tools, monkeypatch):
        monkeypatch.setattr(
            apps_module.subprocess, "run", lambda cmd, **kw: FakeCompleted()
        )
        assert "no output" in tools.launch("ls").content

    def test_long_output_is_truncated(self, tools, monkeypatch):
        monkeypatch.setattr(
            apps_module.subprocess, "run",
            lambda cmd, **kw: FakeCompleted(stdout="x" * 9000),
        )
        assert "(truncated)" in tools.launch("ls").content

    def test_a_timeout_is_reported(self, tools, monkeypatch):
        def raise_timeout(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 5)

        monkeypatch.setattr(apps_module.subprocess, "run", raise_timeout)
        assert "timed out" in tools.launch("ls").content

    def test_a_timeout_is_always_set(self, tools, run_calls):
        tools.launch("ls")
        assert run_calls[0].get("timeout") is not None

    def test_an_os_error_is_reported(self, tools, monkeypatch):
        def raise_oserror(cmd, **kwargs):
            raise OSError("exec format error")

        monkeypatch.setattr(apps_module.subprocess, "run", raise_oserror)
        assert tools.launch("ls").ok is False


class TestGuiApplications:
    def test_gui_apps_are_detached(self, tools, popen_calls, run_calls):
        result = tools.launch("firefox")
        assert "Launched firefox" in result.content
        assert popen_calls[0]["start_new_session"] is True
        assert run_calls == []

    def test_gui_output_is_discarded(self, tools, popen_calls):
        tools.launch("firefox")
        assert popen_calls[0]["stdout"] is subprocess.DEVNULL
        assert popen_calls[0]["stderr"] is subprocess.DEVNULL

    def test_gui_args_are_forwarded(self, tools, popen_calls):
        tools.launch("firefox", ["https://example.com"])
        assert popen_calls[0]["cmd"] == ["/usr/bin/firefox", "https://example.com"]

    def test_a_non_gui_command_is_not_detached(self, tools, popen_calls, run_calls):
        tools.launch("ls")
        assert popen_calls == []


class TestToolSpec:
    def test_registered_and_namespaced(self, on_path):
        registry = ToolRegistry()
        registry.register_all(build_tools(["ls"]))
        assert registry.names() == ["app__launch"]

    def test_never_counts_as_a_safe_call(self, on_path):
        """Whatever the whitelist allows runs outside the filesystem sandbox."""
        assert build_tools(["ls"])[0].risk is Risk.DESTRUCTIVE

    def test_the_description_points_at_the_sandboxed_tools(self, on_path):
        assert "fs__" in build_tools(["ls"])[0].description

    def test_an_unlisted_command_is_refused_up_front(self, on_path):
        """Refused before the user is asked, not after they say yes."""
        precheck = build_tools(["ls"])[0].precheck
        assert "not in the command whitelist" in precheck({"application": "rm -rf /"})

    def test_a_listed_command_passes_the_precheck(self, on_path):
        assert build_tools(["ls"])[0].precheck({"application": "ls -la"}) is None

    def test_an_unparseable_command_is_refused_up_front(self, on_path):
        assert build_tools(["ls"])[0].precheck({"application": 'ls "unclosed'})

    async def test_dispatch_through_the_registry(self, on_path, run_calls):
        registry = ToolRegistry()
        registry.register_all(build_tools(["ls"]))
        assert (await registry.call("app__launch", {"application": "ls"})).ok is True


class TestApprovalScope:
    def test_the_scope_is_the_whole_command_line(self, on_path):
        scope_for = build_tools(["ls"])[0].scope_for
        assert scope_for({"application": "ls", "args": ["-la", "/tmp"]}) == "ls -la /tmp"

    def test_a_single_string_arg_is_handled(self, on_path):
        scope_for = build_tools(["echo"])[0].scope_for
        assert scope_for({"application": "echo", "args": "hello"}) == "echo hello"

    def test_no_args_gives_the_bare_command(self, on_path):
        assert build_tools(["ls"])[0].scope_for({"application": "ls"}) == "ls"

    def test_different_arguments_are_different_scopes(self, on_path):
        """Approving "ls -la" must not approve "rm -rf"."""
        scope_for = build_tools(["ls"])[0].scope_for
        assert scope_for({"application": "ls", "args": ["-la"]}) != \
            scope_for({"application": "ls", "args": ["-l"]})


class TestDefaults:
    def test_the_default_whitelist_holds_no_file_manipulation_tools(self, clean_env):
        """rm, cat, mkdir and touch reach the whole disk: they never pass
        through the path sandbox, so they do not belong in the default."""
        from config import Settings

        whitelist = Settings(_env_file=None).command_whitelist_list
        for dangerous in ("rm", "cat", "mkdir", "touch", "ls"):
            assert dangerous not in whitelist
