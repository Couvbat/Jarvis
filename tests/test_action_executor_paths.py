"""Sandbox enforcement in ActionExecutor._is_path_allowed."""

from pathlib import Path

from action_executor import ActionExecutor


class TestPathValidation:
    def test_path_inside_sandbox_is_allowed(self, executor, sandbox):
        assert executor._is_path_allowed(sandbox / "notes.txt") is True

    def test_nested_path_is_allowed(self, executor, sandbox):
        assert executor._is_path_allowed(sandbox / "a" / "b" / "c.txt") is True

    def test_sandbox_root_itself_is_allowed(self, executor, sandbox):
        assert executor._is_path_allowed(sandbox) is True

    def test_path_outside_sandbox_is_denied(self, executor, tmp_path):
        assert executor._is_path_allowed(tmp_path / "outside.txt") is False

    def test_absolute_system_path_is_denied(self, executor):
        assert executor._is_path_allowed(Path("/etc/passwd")) is False

    def test_dotdot_traversal_is_denied(self, executor, sandbox):
        assert executor._is_path_allowed(sandbox / ".." / "escape.txt") is False

    def test_deep_dotdot_traversal_is_denied(self, executor, sandbox):
        target = sandbox / "sub" / ".." / ".." / ".." / "etc" / "passwd"
        assert executor._is_path_allowed(target) is False

    def test_symlink_escaping_the_sandbox_is_denied(self, executor, sandbox, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("classified")
        link = sandbox / "link.txt"
        link.symlink_to(outside)
        assert executor._is_path_allowed(link) is False

    def test_sibling_with_shared_prefix_is_denied(self, executor, sandbox):
        """``/x/sandbox-evil`` must not pass because it starts with ``/x/sandbox``."""
        sibling = sandbox.parent / (sandbox.name + "-evil")
        sibling.mkdir()
        assert executor._is_path_allowed(sibling / "f.txt") is False

    def test_multiple_allowed_directories(self, settings, tmp_path, approve_all):
        first = tmp_path / "one"
        second = tmp_path / "two"
        first.mkdir()
        second.mkdir()
        settings.allowed_directories = f"{first},{second}"
        instance = ActionExecutor(confirmation_callback=approve_all)
        assert instance._is_path_allowed(first / "a.txt") is True
        assert instance._is_path_allowed(second / "b.txt") is True
        assert instance._is_path_allowed(tmp_path / "c.txt") is False


class TestSandboxedOperations:
    def test_create_outside_sandbox_is_refused(self, executor, tmp_path):
        target = tmp_path / "escaped.txt"
        result = executor.execute_file_operation("create_file", str(target), "x")
        assert "not in allowed directories" in result
        assert not target.exists()

    def test_read_outside_sandbox_is_refused(self, executor, tmp_path):
        target = tmp_path / "secret.txt"
        target.write_text("classified")
        result = executor.execute_file_operation("read_file", str(target))
        assert "not in allowed directories" in result
        assert "classified" not in result

    def test_delete_outside_sandbox_is_refused(self, executor, tmp_path):
        target = tmp_path / "victim.txt"
        target.write_text("keep me")
        result = executor.execute_file_operation("delete_file", str(target))
        assert "not in allowed directories" in result
        assert target.exists()

    def test_sandbox_check_precedes_confirmation(self, executor, tmp_path, approve_all):
        """A path outside the sandbox is rejected without ever asking the user."""
        executor.execute_file_operation("delete_file", str(tmp_path / "x.txt"))
        assert approve_all.calls == []

    def test_tilde_is_expanded_before_validation(self, executor, monkeypatch, sandbox):
        monkeypatch.setenv("HOME", str(sandbox))
        result = executor.execute_file_operation("create_file", "~/from-tilde.txt", "hi")
        assert "File created" in result
        assert (sandbox / "from-tilde.txt").read_text() == "hi"
