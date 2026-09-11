"""File operations in ActionExecutor.execute_file_operation."""

import pytest


class TestCreateFile:
    def test_creates_file_with_content(self, executor, sandbox):
        result = executor.execute_file_operation(
            "create_file", str(sandbox / "notes.txt"), "hello"
        )
        assert "File created" in result
        assert (sandbox / "notes.txt").read_text() == "hello"

    def test_creates_empty_file_when_content_is_none(self, executor, sandbox):
        executor.execute_file_operation("create_file", str(sandbox / "empty.txt"))
        assert (sandbox / "empty.txt").read_text() == ""

    def test_creates_missing_parent_directories(self, executor, sandbox):
        executor.execute_file_operation(
            "create_file", str(sandbox / "a" / "b" / "c.txt"), "deep"
        )
        assert (sandbox / "a" / "b" / "c.txt").read_text() == "deep"

    def test_overwrites_existing_file(self, executor, sandbox):
        target = sandbox / "notes.txt"
        target.write_text("original")
        executor.execute_file_operation("create_file", str(target), "replaced")
        assert target.read_text() == "replaced"

    def test_requires_confirmation(self, executor, sandbox, approve_all):
        executor.execute_file_operation("create_file", str(sandbox / "x.txt"), "x")
        assert len(approve_all.calls) == 1
        description, item = approve_all.calls[0]
        assert "create_file" in description
        assert item == f"create_file:{sandbox}"

    def test_denied_confirmation_creates_nothing(self, executor, sandbox, deny_all):
        executor.confirmation_callback = deny_all
        result = executor.execute_file_operation(
            "create_file", str(sandbox / "x.txt"), "x"
        )
        assert "cancelled by user" in result
        assert not (sandbox / "x.txt").exists()

    def test_no_callback_means_denied(self, executor, sandbox):
        executor.confirmation_callback = None
        result = executor.execute_file_operation(
            "create_file", str(sandbox / "x.txt"), "x"
        )
        assert "cancelled by user" in result
        assert not (sandbox / "x.txt").exists()


class TestReadFile:
    def test_reads_content(self, executor, sandbox):
        (sandbox / "notes.txt").write_text("the content")
        result = executor.execute_file_operation("read_file", str(sandbox / "notes.txt"))
        assert "the content" in result

    def test_missing_file_reports_error(self, executor, sandbox):
        result = executor.execute_file_operation("read_file", str(sandbox / "nope.txt"))
        assert "Error" in result and "not found" in result

    def test_directory_is_not_readable_as_a_file(self, executor, sandbox):
        (sandbox / "adir").mkdir()
        result = executor.execute_file_operation("read_file", str(sandbox / "adir"))
        assert "Not a file" in result

    def test_long_content_is_truncated(self, executor, sandbox):
        (sandbox / "big.txt").write_text("x" * 5000)
        result = executor.execute_file_operation("read_file", str(sandbox / "big.txt"))
        assert "(truncated)" in result
        assert result.count("x") == 1000

    def test_binary_file_reports_error_instead_of_raising(self, executor, sandbox):
        (sandbox / "blob.bin").write_bytes(b"\xff\xfe\x00\x80binary")
        result = executor.execute_file_operation("read_file", str(sandbox / "blob.bin"))
        assert result.startswith("Error")


class TestDeleteFile:
    def test_deletes_file(self, executor, sandbox):
        target = sandbox / "gone.txt"
        target.write_text("bye")
        result = executor.execute_file_operation("delete_file", str(target))
        assert "File deleted" in result
        assert not target.exists()

    def test_missing_file_reports_error(self, executor, sandbox):
        result = executor.execute_file_operation("delete_file", str(sandbox / "nope.txt"))
        assert "Error" in result and "not found" in result

    def test_directory_is_refused(self, executor, sandbox):
        target = sandbox / "adir"
        target.mkdir()
        result = executor.execute_file_operation("delete_file", str(target))
        assert "Not a file" in result
        assert target.exists()

    def test_requires_confirmation(self, executor, sandbox, approve_all):
        target = sandbox / "gone.txt"
        target.write_text("bye")
        executor.execute_file_operation("delete_file", str(target))
        assert approve_all.calls[0][1] == f"delete_file:{sandbox}"

    def test_denied_confirmation_keeps_the_file(self, executor, sandbox, deny_all):
        executor.confirmation_callback = deny_all
        target = sandbox / "keep.txt"
        target.write_text("still here")
        result = executor.execute_file_operation("delete_file", str(target))
        assert "cancelled by user" in result
        assert target.exists()


class TestDirectoryOperations:
    def test_create_directory(self, executor, sandbox):
        result = executor.execute_file_operation(
            "create_directory", str(sandbox / "newdir")
        )
        assert "Directory created" in result
        assert (sandbox / "newdir").is_dir()

    def test_create_directory_is_idempotent(self, executor, sandbox):
        (sandbox / "newdir").mkdir()
        result = executor.execute_file_operation(
            "create_directory", str(sandbox / "newdir")
        )
        assert "Directory created" in result

    def test_list_directory_shows_files_and_dirs(self, executor, sandbox):
        (sandbox / "a.txt").write_text("a")
        (sandbox / "subdir").mkdir()
        result = executor.execute_file_operation("list_directory", str(sandbox))
        assert "a.txt" in result
        assert "subdir" in result

    def test_list_directory_puts_directories_first(self, executor, sandbox):
        (sandbox / "a.txt").write_text("a")
        (sandbox / "zdir").mkdir()
        result = executor.execute_file_operation("list_directory", str(sandbox))
        assert result.index("zdir") < result.index("a.txt")

    def test_list_missing_directory_reports_error(self, executor, sandbox):
        result = executor.execute_file_operation("list_directory", str(sandbox / "nope"))
        assert "Error" in result and "not found" in result

    def test_list_of_a_file_reports_error(self, executor, sandbox):
        (sandbox / "a.txt").write_text("a")
        result = executor.execute_file_operation("list_directory", str(sandbox / "a.txt"))
        assert "Not a directory" in result

    def test_long_listing_is_truncated(self, executor, sandbox):
        for index in range(60):
            (sandbox / f"file{index:03d}.txt").write_text("x")
        result = executor.execute_file_operation("list_directory", str(sandbox))
        assert "(truncated)" in result
        assert len(result.splitlines()) == 52  # header + 50 entries + marker


class TestUnknownOperation:
    def test_unknown_operation_reports_error(self, executor, sandbox):
        result = executor.execute_file_operation("chmod", str(sandbox / "a.txt"))
        assert "Unknown operation" in result

    def test_unknown_operation_is_not_confirmed(self, executor, sandbox, approve_all):
        executor.execute_file_operation("chmod", str(sandbox / "a.txt"))
        assert approve_all.calls == []


class TestWhitelistIntegration:
    def test_approving_with_whitelist_persists_the_directory(
        self, executor, sandbox, approve_and_whitelist
    ):
        executor.confirmation_callback = approve_and_whitelist
        executor.execute_file_operation("create_file", str(sandbox / "a.txt"), "a")
        assert executor.whitelist_manager.is_whitelisted(
            "file_operations", f"create_file:{sandbox}"
        )

    def test_whitelisted_directory_skips_later_prompts(
        self, executor, sandbox, approve_and_whitelist
    ):
        executor.confirmation_callback = approve_and_whitelist
        executor.execute_file_operation("create_file", str(sandbox / "a.txt"), "a")
        executor.execute_file_operation("create_file", str(sandbox / "b.txt"), "b")
        assert len(approve_and_whitelist.calls) == 1
        assert (sandbox / "b.txt").exists()

    def test_whitelist_is_scoped_to_the_operation(
        self, executor, sandbox, approve_and_whitelist
    ):
        executor.confirmation_callback = approve_and_whitelist
        executor.execute_file_operation("create_file", str(sandbox / "a.txt"), "a")
        executor.execute_file_operation("delete_file", str(sandbox / "a.txt"))
        assert len(approve_and_whitelist.calls) == 2

    def test_whitelist_is_scoped_to_the_parent_directory(
        self, executor, sandbox, approve_and_whitelist
    ):
        executor.confirmation_callback = approve_and_whitelist
        executor.execute_file_operation("create_file", str(sandbox / "a.txt"), "a")
        executor.execute_file_operation(
            "create_file", str(sandbox / "sub" / "b.txt"), "b"
        )
        assert len(approve_and_whitelist.calls) == 2


# --------------------------------------------------------------------------- #
# Known gaps - these turn green the day the behaviour is fixed.
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason="BUG-01: read_file bypasses confirmation")
def test_read_file_should_require_confirmation(executor, sandbox, approve_all):
    """README: 'All file operations ... require user approval'.

    ``read_file`` asks for nothing, so any prompt injection that reaches the
    LLM can exfiltrate readable files inside ALLOWED_DIRECTORIES.
    """
    (sandbox / "secret.txt").write_text("token=abc")
    executor.execute_file_operation("read_file", str(sandbox / "secret.txt"))
    assert len(approve_all.calls) == 1


@pytest.mark.xfail(strict=True, reason="BUG-01: list_directory bypasses confirmation")
def test_list_directory_should_require_confirmation(executor, sandbox, approve_all):
    executor.execute_file_operation("list_directory", str(sandbox))
    assert len(approve_all.calls) == 1


@pytest.mark.xfail(strict=True, reason="BUG-01: create_directory bypasses confirmation")
def test_create_directory_should_require_confirmation(executor, sandbox, approve_all):
    executor.execute_file_operation("create_directory", str(sandbox / "newdir"))
    assert len(approve_all.calls) == 1


@pytest.mark.xfail(strict=True, reason="BUG-03: delete_directory is not implemented")
def test_delete_directory_operation_should_exist(executor, sandbox):
    """``delete_file`` tells the user to 'use delete_directory' - which does not exist."""
    target = sandbox / "adir"
    target.mkdir()
    result = executor.execute_file_operation("delete_directory", str(target))
    assert "Unknown operation" not in result
    assert not target.exists()


@pytest.mark.xfail(strict=True, reason="BUG-05: no edit/append operation")
def test_append_to_file_should_exist(executor, sandbox):
    """The system prompt advertises editing files; only whole-file writes exist."""
    target = sandbox / "log.txt"
    target.write_text("line1\n")
    executor.execute_file_operation("append_file", str(target), "line2\n")
    assert target.read_text() == "line1\nline2\n"
