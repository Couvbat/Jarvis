"""Tests for the path sandbox and denied patterns (policy/paths.py)."""

from pathlib import Path

import pytest

from policy.paths import PathPolicy, PathVerdict


@pytest.fixture
def root(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return sandbox


@pytest.fixture
def policy(root):
    return PathPolicy([root], denied_patterns=[".ssh", ".env", "*_history", "id_rsa*"])


class TestSandbox:
    def test_a_path_inside_is_allowed(self, policy, root):
        assert policy.check(root / "notes.txt").allowed is True

    def test_a_nested_path_is_allowed(self, policy, root):
        assert policy.check(root / "a" / "b" / "c.txt").allowed is True

    def test_the_root_itself_is_allowed(self, policy, root):
        assert policy.check(root).allowed is True

    def test_a_path_outside_is_refused(self, policy, tmp_path):
        verdict = policy.check(tmp_path / "outside.txt")
        assert verdict.allowed is False
        assert "outside the allowed directories" in verdict.reason

    def test_a_system_path_is_refused(self, policy):
        assert policy.check("/etc/passwd").allowed is False

    def test_dotdot_traversal_is_refused(self, policy, root):
        assert policy.check(root / ".." / "escape.txt").allowed is False

    def test_deep_traversal_is_refused(self, policy, root):
        assert policy.check(root / "a" / ".." / ".." / ".." / "etc" / "passwd").allowed is False

    def test_a_symlink_pointing_out_is_refused(self, policy, root, tmp_path):
        """Resolution follows symlinks, so a link out of the sandbox is caught."""
        secret = tmp_path / "secret.txt"
        secret.write_text("classified")
        link = root / "link.txt"
        link.symlink_to(secret)
        assert policy.check(link).allowed is False

    def test_a_sibling_sharing_a_prefix_is_refused(self, policy, root):
        sibling = root.parent / (root.name + "-evil")
        sibling.mkdir()
        assert policy.check(sibling / "f.txt").allowed is False

    def test_several_roots_are_honoured(self, tmp_path):
        first, second = tmp_path / "one", tmp_path / "two"
        first.mkdir()
        second.mkdir()
        policy = PathPolicy([first, second])
        assert policy.check(first / "a").allowed is True
        assert policy.check(second / "b").allowed is True
        assert policy.check(tmp_path / "c").allowed is False

    def test_a_path_need_not_exist(self, policy, root):
        """Creating a file means checking a path that is not there yet."""
        assert policy.check(root / "brand" / "new.txt").allowed is True

    def test_tilde_is_expanded(self, monkeypatch, root):
        monkeypatch.setenv("HOME", str(root))
        policy = PathPolicy(["~"])
        assert policy.check("~/inside.txt").allowed is True

    def test_an_empty_root_entry_is_ignored(self, root):
        """An empty ALLOWED_DIRECTORIES entry must not silently allow the CWD."""
        policy = PathPolicy([root, "", "  "])
        assert policy.allowed_directories == [root.resolve()]

    def test_no_roots_allows_nothing(self, tmp_path):
        policy = PathPolicy([])
        assert policy.check(tmp_path / "a.txt").allowed is False


class TestDeniedPatterns:
    def test_a_denied_directory_component_is_refused(self, policy, root):
        verdict = policy.check(root / ".ssh" / "id_rsa")
        assert verdict.allowed is False
        assert ".ssh" in verdict.reason

    def test_a_denied_file_name_is_refused(self, policy, root):
        assert policy.check(root / ".env").allowed is False

    def test_a_glob_pattern_matches(self, policy, root):
        assert policy.check(root / ".bash_history").allowed is False
        assert policy.check(root / "id_rsa.pub").allowed is False

    def test_a_denied_name_deep_in_the_tree_is_refused(self, policy, root):
        assert policy.check(root / "projects" / "app" / ".env").allowed is False

    def test_a_similar_but_different_name_is_allowed(self, policy, root):
        assert policy.check(root / "environment.txt").allowed is True
        assert policy.check(root / "history.md").allowed is True

    def test_patterns_are_only_applied_below_the_allowed_root(self, tmp_path):
        """Allowing a sensitive directory on purpose still works.

        The user asked for this root explicitly; the denied list is there to
        stop the model wandering into secrets, not to override that choice.
        """
        root = tmp_path / ".ssh"
        root.mkdir()
        policy = PathPolicy([root], denied_patterns=[".ssh"])
        assert policy.check(root / "config").allowed is True

    def test_the_sandbox_is_checked_before_the_denied_list(self, policy, tmp_path):
        verdict = policy.check(tmp_path / ".ssh" / "id_rsa")
        assert "outside the allowed directories" in verdict.reason

    def test_no_patterns_means_nothing_is_denied(self, root):
        policy = PathPolicy([root], denied_patterns=[])
        assert policy.check(root / ".ssh" / "id_rsa").allowed is True

    def test_blank_patterns_are_dropped(self, root):
        policy = PathPolicy([root], denied_patterns=["", "  ", ".ssh"])
        assert policy.denied_patterns == [".ssh"]


class TestVerdict:
    def test_verdict_is_truthy_when_allowed(self, policy, root):
        assert bool(policy.check(root / "a.txt")) is True

    def test_verdict_is_falsy_when_refused(self, policy):
        assert bool(policy.check("/etc/passwd")) is False

    def test_verdict_carries_the_resolved_path(self, policy, root):
        verdict = policy.check(root / "sub" / ".." / "a.txt")
        assert verdict.path == root / "a.txt"

    def test_an_unusable_path_is_refused_not_raised(self, policy):
        verdict = policy.check("\x00invalid")
        assert isinstance(verdict, PathVerdict)
        assert verdict.allowed is False


class TestFromSettings:
    def test_reads_the_configured_directories(self, settings, tmp_path):
        settings.allowed_directories = str(tmp_path)
        settings.denied_patterns = ".ssh"
        policy = PathPolicy.from_settings()
        assert policy.allowed_directories == [tmp_path.resolve()]
        assert policy.denied_patterns == [".ssh"]

    def test_defaults_do_not_include_the_whole_home_tree(self, clean_env):
        """Allowing "/home" would put every household dotfile in the sandbox."""
        from config import Settings

        fresh = Settings(_env_file=None)
        assert Path("/home") not in fresh.allowed_dirs_list

    def test_credentials_are_denied_by_default(self, clean_env):
        from config import Settings

        patterns = Settings(_env_file=None).denied_patterns_list
        for expected in (".ssh", ".aws", ".env", "*_history"):
            assert expected in patterns
