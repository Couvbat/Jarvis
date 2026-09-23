"""Tests for the filesystem tools (tools/local/filesystem.py)."""

from pathlib import Path

import pytest

from jarvis.policy.paths import PathPolicy
from jarvis.tools.local import filesystem
from jarvis.tools.local.filesystem import FilesystemTools, build_tools
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.schema import Risk


@pytest.fixture
def root(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return sandbox


@pytest.fixture
def fs(root):
    return FilesystemTools(PathPolicy([root], denied_patterns=[".ssh", ".env"]))


@pytest.fixture
def registry(root):
    instance = ToolRegistry()
    instance.register_all(
        build_tools(PathPolicy([root], denied_patterns=[".ssh", ".env"]))
    )
    return instance


class TestArgumentCoercion:
    def test_a_boolean_is_not_a_number(self, fs, root):
        (root / "a.txt").write_text("x")
        result = fs.read(root / "a.txt", start_line=True)
        assert result.ok is False
        assert "must be a number" in result.content

    @pytest.mark.parametrize("value,expected", [
        (True, True), (False, False), (1, True), (0, False),
        ("true", True), ("YES", True), ("no", False), ("", False), (None, False),
    ])
    def test_boolean_coercion(self, value, expected):
        assert filesystem._as_bool(value, default=False) is expected

    def test_an_empty_path_is_reported(self, fs):
        result = fs.read("")
        assert result.ok is False
        assert "path is required" in result.content

    def test_a_missing_path_is_reported(self, fs):
        assert fs.read(None).ok is False

    def test_an_unreadable_file_is_reported_not_raised(self, fs, root, monkeypatch):
        target = root / "a.txt"
        target.write_text("x")

        def explode(*args, **kwargs):
            raise OSError("I/O error")

        monkeypatch.setattr(Path, "read_text", explode)
        result = fs.read(target)
        assert result.ok is False
        assert "could not read" in result.content


class TestRead:
    def test_reads_a_file(self, fs, root):
        (root / "a.txt").write_text("hello\nworld\n")
        result = fs.read(root / "a.txt")
        assert result.ok is True
        assert "hello" in result.content and "world" in result.content

    def test_reports_the_range_and_total(self, fs, root):
        (root / "a.txt").write_text("\n".join(str(n) for n in range(100)))
        assert "of 100" in fs.read(root / "a.txt").content

    def test_reads_a_line_range(self, fs, root):
        (root / "a.txt").write_text("\n".join(f"line{n}" for n in range(50)))
        result = fs.read(root / "a.txt", start_line=10, max_lines=3)
        assert "line9" in result.content and "line11" in result.content
        assert "line12" not in result.content

    def test_caps_the_number_of_lines(self, fs, root):
        (root / "big.txt").write_text("\n".join("x" for _ in range(5000)))
        body = fs.read(root / "big.txt").content.splitlines()
        assert len(body) == filesystem.DEFAULT_MAX_LINES + 1  # header + lines

    def test_clips_very_long_lines(self, fs, root):
        (root / "wide.txt").write_text("y" * 5000)
        assert len(fs.read(root / "wide.txt").content.splitlines()[1]) < 1100

    def test_string_arguments_are_accepted(self, fs, root):
        """Local models routinely emit "10" where an integer is wanted."""
        (root / "a.txt").write_text("\n".join(f"line{n}" for n in range(50)))
        result = fs.read(root / "a.txt", start_line="10", max_lines="2")
        assert result.ok is True
        assert "line9" in result.content

    def test_an_unusable_argument_is_reported(self, fs, root):
        (root / "a.txt").write_text("x")
        result = fs.read(root / "a.txt", start_line="soon")
        assert result.ok is False
        assert "start_line" in result.content

    def test_a_start_beyond_the_end_is_not_an_error(self, fs, root):
        (root / "a.txt").write_text("one\ntwo\n")
        result = fs.read(root / "a.txt", start_line=99)
        assert result.ok is True
        assert "nothing at line 99" in result.content

    def test_missing_file(self, fs, root):
        assert fs.read(root / "nope.txt").ok is False

    def test_a_directory_is_not_a_file(self, fs, root):
        (root / "d").mkdir()
        assert "not a file" in fs.read(root / "d").content

    def test_binary_files_are_refused(self, fs, root):
        (root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        assert "binary" in fs.read(root / "blob.bin").content

    def test_outside_the_sandbox(self, fs, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("classified")
        result = fs.read(secret)
        assert result.ok is False
        assert "classified" not in result.content

    def test_a_denied_name_is_refused(self, fs, root):
        (root / ".env").write_text("TOKEN=abc")
        result = fs.read(root / ".env")
        assert result.ok is False
        assert "TOKEN" not in result.content


class TestList:
    def test_lists_entries(self, fs, root):
        (root / "a.txt").write_text("a")
        (root / "sub").mkdir()
        content = fs.list_directory(root).content
        assert "a.txt" in content and "sub/" in content

    def test_directories_come_first(self, fs, root):
        (root / "a.txt").write_text("a")
        (root / "zdir").mkdir()
        content = fs.list_directory(root).content
        assert content.index("zdir/") < content.index("a.txt")

    def test_file_sizes_are_shown(self, fs, root):
        (root / "a.txt").write_text("12345")
        assert "5 bytes" in fs.list_directory(root).content

    def test_an_empty_directory_says_so(self, fs, root):
        assert "(empty)" in fs.list_directory(root).content

    def test_a_long_listing_is_capped(self, fs, root):
        for index in range(filesystem.MAX_LIST_ENTRIES + 20):
            (root / f"f{index:04d}.txt").write_text("x")
        assert "and 20 more" in fs.list_directory(root).content

    def test_a_file_is_not_a_directory(self, fs, root):
        (root / "a.txt").write_text("a")
        assert "not a directory" in fs.list_directory(root / "a.txt").content


class TestStat:
    def test_describes_a_file(self, fs, root):
        (root / "a.txt").write_text("12345")
        content = fs.stat(root / "a.txt").content
        assert "type: file" in content and "size: 5 bytes" in content

    def test_describes_a_directory(self, fs, root):
        (root / "d").mkdir()
        assert "type: directory" in fs.stat(root / "d").content

    def test_missing_path(self, fs, root):
        assert fs.stat(root / "nope").ok is False


class TestSearch:
    def test_finds_by_name_pattern(self, fs, root):
        (root / "a.py").write_text("x")
        (root / "b.txt").write_text("x")
        content = fs.search(root, name_pattern="*.py").content
        assert "a.py" in content and "b.txt" not in content

    def test_searches_subdirectories(self, fs, root):
        (root / "sub").mkdir()
        (root / "sub" / "deep.py").write_text("x")
        assert "deep.py" in fs.search(root, name_pattern="*.py").content

    def test_finds_by_content(self, fs, root):
        (root / "a.txt").write_text("nothing here")
        (root / "b.txt").write_text("first\nthe needle is here\n")
        content = fs.search(root, content_pattern="needle").content
        assert "b.txt:2" in content
        assert "a.txt" not in content

    def test_content_search_is_case_insensitive(self, fs, root):
        (root / "a.txt").write_text("The NEEDLE")
        assert "a.txt" in fs.search(root, content_pattern="needle").content

    def test_name_and_content_combine(self, fs, root):
        (root / "a.py").write_text("needle")
        (root / "b.txt").write_text("needle")
        content = fs.search(root, name_pattern="*.py", content_pattern="needle").content
        assert "a.py" in content and "b.txt" not in content

    def test_no_match_says_so(self, fs, root):
        (root / "a.txt").write_text("x")
        assert "No match" in fs.search(root, content_pattern="absent").content

    def test_at_least_one_criterion_is_required(self, fs, root):
        assert fs.search(root).ok is False

    def test_results_are_capped(self, fs, root):
        for index in range(30):
            (root / f"f{index}.txt").write_text("needle")
        content = fs.search(root, content_pattern="needle", max_results=5).content
        assert content.count("needle") <= 6  # 5 results plus the trailing note

    def test_denied_names_do_not_surface(self, fs, root):
        """Search must not become a way around the denied list."""
        (root / ".env").write_text("TOKEN=secret")
        content = fs.search(root, content_pattern="TOKEN").content
        assert ".env" not in content

    def test_binary_files_are_skipped_for_content_search(self, fs, root):
        (root / "blob.bin").write_bytes(b"needle\x00\x01")
        assert "No match" in fs.search(root, content_pattern="needle").content


class TestWrite:
    def test_create(self, fs, root):
        result = fs.write(root / "a.txt", "hello", mode="create")
        assert result.ok is True
        assert (root / "a.txt").read_text() == "hello"

    def test_create_makes_parent_directories(self, fs, root):
        fs.write(root / "a" / "b" / "c.txt", "deep")
        assert (root / "a" / "b" / "c.txt").read_text() == "deep"

    def test_create_refuses_to_clobber(self, fs, root):
        (root / "a.txt").write_text("original")
        result = fs.write(root / "a.txt", "new", mode="create")
        assert result.ok is False
        assert (root / "a.txt").read_text() == "original"

    def test_overwrite(self, fs, root):
        (root / "a.txt").write_text("original")
        fs.write(root / "a.txt", "replaced", mode="overwrite")
        assert (root / "a.txt").read_text() == "replaced"

    def test_append(self, fs, root):
        (root / "a.txt").write_text("line1\n")
        fs.write(root / "a.txt", "line2\n", mode="append")
        assert (root / "a.txt").read_text() == "line1\nline2\n"

    def test_append_creates_a_missing_file(self, fs, root):
        fs.write(root / "new.txt", "x", mode="append")
        assert (root / "new.txt").read_text() == "x"

    def test_default_mode_is_create(self, fs, root):
        (root / "a.txt").write_text("original")
        assert fs.write(root / "a.txt", "new").ok is False

    def test_an_unknown_mode_is_reported(self, fs, root):
        result = fs.write(root / "a.txt", "x", mode="replace")
        assert result.ok is False
        assert "create, overwrite or append" in result.content

    def test_outside_the_sandbox(self, fs, tmp_path):
        result = fs.write(tmp_path / "escaped.txt", "x")
        assert result.ok is False
        assert not (tmp_path / "escaped.txt").exists()


class TestEdit:
    def test_replaces_a_unique_match(self, fs, root):
        (root / "a.txt").write_text("hello world\n")
        assert fs.edit(root / "a.txt", "world", "there").ok is True
        assert (root / "a.txt").read_text() == "hello there\n"

    def test_an_absent_match_is_reported(self, fs, root):
        (root / "a.txt").write_text("hello\n")
        result = fs.edit(root / "a.txt", "absent", "x")
        assert result.ok is False
        assert (root / "a.txt").read_text() == "hello\n"

    def test_an_ambiguous_match_changes_nothing(self, fs, root):
        """Picking one of several matches arbitrarily would be unreviewable."""
        (root / "a.txt").write_text("x\nx\n")
        result = fs.edit(root / "a.txt", "x", "y")
        assert result.ok is False
        assert "appears 2 times" in result.content
        assert (root / "a.txt").read_text() == "x\nx\n"

    def test_an_empty_replacement_deletes_the_text(self, fs, root):
        (root / "a.txt").write_text("keep REMOVE keep")
        fs.edit(root / "a.txt", " REMOVE", "")
        assert (root / "a.txt").read_text() == "keep keep"

    def test_multiline_replacement(self, fs, root):
        (root / "a.txt").write_text("start\nold1\nold2\nend\n")
        fs.edit(root / "a.txt", "old1\nold2", "new")
        assert (root / "a.txt").read_text() == "start\nnew\nend\n"

    def test_old_text_is_required(self, fs, root):
        (root / "a.txt").write_text("x")
        assert fs.edit(root / "a.txt", "", "y").ok is False

    def test_a_missing_file_is_reported(self, fs, root):
        assert fs.edit(root / "nope.txt", "a", "b").ok is False


class TestMkdir:
    def test_creates_a_directory(self, fs, root):
        assert fs.make_directory(root / "new").ok is True
        assert (root / "new").is_dir()

    def test_creates_parents(self, fs, root):
        fs.make_directory(root / "a" / "b" / "c")
        assert (root / "a" / "b" / "c").is_dir()

    def test_is_idempotent(self, fs, root):
        fs.make_directory(root / "new")
        assert fs.make_directory(root / "new").ok is True

    def test_refuses_over_an_existing_file(self, fs, root):
        (root / "a.txt").write_text("x")
        assert fs.make_directory(root / "a.txt").ok is False


class TestDelete:
    def test_deletes_a_file(self, fs, root):
        (root / "a.txt").write_text("x")
        assert fs.delete(root / "a.txt").ok is True
        assert not (root / "a.txt").exists()

    def test_deletes_an_empty_directory(self, fs, root):
        (root / "d").mkdir()
        assert fs.delete(root / "d").ok is True

    def test_a_non_empty_directory_needs_recursive(self, fs, root):
        (root / "d").mkdir()
        (root / "d" / "a.txt").write_text("x")
        result = fs.delete(root / "d")
        assert result.ok is False
        assert "recursive=true" in result.content
        assert (root / "d" / "a.txt").exists()

    def test_recursive_deletes_the_tree(self, fs, root):
        (root / "d" / "sub").mkdir(parents=True)
        (root / "d" / "sub" / "a.txt").write_text("x")
        assert fs.delete(root / "d", recursive=True).ok is True
        assert not (root / "d").exists()

    def test_recursive_accepts_a_string(self, fs, root):
        (root / "d").mkdir()
        (root / "d" / "a.txt").write_text("x")
        assert fs.delete(root / "d", recursive="true").ok is True

    def test_refuses_to_delete_the_sandbox_root(self, fs, root):
        result = fs.delete(root, recursive=True)
        assert result.ok is False
        assert root.exists()

    def test_missing_path(self, fs, root):
        assert fs.delete(root / "nope").ok is False

    def test_outside_the_sandbox(self, fs, tmp_path):
        victim = tmp_path / "victim.txt"
        victim.write_text("keep")
        assert fs.delete(victim).ok is False
        assert victim.exists()


class TestMoveAndCopy:
    def test_move_renames(self, fs, root):
        (root / "a.txt").write_text("x")
        assert fs.move(root / "a.txt", root / "b.txt").ok is True
        assert (root / "b.txt").read_text() == "x"
        assert not (root / "a.txt").exists()

    def test_move_creates_the_destination_parent(self, fs, root):
        (root / "a.txt").write_text("x")
        fs.move(root / "a.txt", root / "sub" / "b.txt")
        assert (root / "sub" / "b.txt").read_text() == "x"

    def test_move_out_of_the_sandbox_is_refused(self, fs, root, tmp_path):
        (root / "a.txt").write_text("x")
        assert fs.move(root / "a.txt", tmp_path / "escaped.txt").ok is False
        assert (root / "a.txt").exists()

    def test_move_in_from_outside_is_refused(self, fs, root, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("x")
        assert fs.move(outside, root / "a.txt").ok is False
        assert outside.exists()

    def test_copy_a_file(self, fs, root):
        (root / "a.txt").write_text("x")
        assert fs.copy(root / "a.txt", root / "b.txt").ok is True
        assert (root / "a.txt").exists() and (root / "b.txt").read_text() == "x"

    def test_copy_a_directory(self, fs, root):
        (root / "d" / "sub").mkdir(parents=True)
        (root / "d" / "sub" / "a.txt").write_text("x")
        fs.copy(root / "d", root / "copy")
        assert (root / "copy" / "sub" / "a.txt").read_text() == "x"

    def test_copy_out_of_the_sandbox_is_refused(self, fs, root, tmp_path):
        (root / "a.txt").write_text("secret")
        assert fs.copy(root / "a.txt", tmp_path / "leak.txt").ok is False
        assert not (tmp_path / "leak.txt").exists()

    def test_a_missing_source_is_reported(self, fs, root):
        assert fs.move(root / "nope", root / "b").ok is False


class TestToolSpecs:
    def test_every_tool_is_namespaced(self, registry):
        assert all(name.startswith("fs__") for name in registry.names())

    def test_the_full_crud_surface_is_present(self, registry):
        assert set(registry.names()) == {
            "fs__read", "fs__list", "fs__stat", "fs__search",
            "fs__write", "fs__edit", "fs__mkdir",
            "fs__delete", "fs__move", "fs__copy",
        }

    @pytest.mark.parametrize("name", ["fs__read", "fs__list", "fs__stat", "fs__search"])
    def test_observing_tools_are_read_only(self, registry, name):
        assert registry.get(name).risk is Risk.READ_ONLY

    def test_delete_is_destructive(self, registry):
        assert registry.get("fs__delete").risk is Risk.DESTRUCTIVE

    def test_creating_a_new_file_is_only_a_write(self, registry, root):
        spec = registry.get("fs__write")
        assert spec.assess({"path": str(root / "new.txt"), "mode": "create"}) is Risk.WRITE

    def test_overwriting_an_existing_file_is_destructive(self, registry, root):
        """Only the arguments can tell creating from clobbering."""
        (root / "a.txt").write_text("original")
        spec = registry.get("fs__write")
        assert spec.assess(
            {"path": str(root / "a.txt"), "mode": "overwrite"}
        ) is Risk.DESTRUCTIVE

    def test_overwriting_a_missing_file_is_only_a_write(self, registry, root):
        spec = registry.get("fs__write")
        assert spec.assess(
            {"path": str(root / "absent.txt"), "mode": "overwrite"}
        ) is Risk.WRITE

    def test_moving_onto_an_existing_file_is_destructive(self, registry, root):
        (root / "b.txt").write_text("victim")
        spec = registry.get("fs__move")
        assert spec.assess(
            {"source": str(root / "a.txt"), "destination": str(root / "b.txt")}
        ) is Risk.DESTRUCTIVE

    def test_every_required_argument_is_in_the_schema(self, registry):
        for spec in registry.specs():
            properties = spec.input_schema["properties"]
            for required in spec.input_schema["required"]:
                assert required in properties, f"{spec.name}: {required}"

    def test_every_tool_has_a_description(self, registry):
        assert all(len(spec.description) > 10 for spec in registry.specs())

    async def test_dispatch_through_the_registry_works(self, registry, root):
        result = await registry.call("fs__write", {"path": str(root / "a.txt"), "content": "hi"})
        assert result.ok is True
        assert (root / "a.txt").read_text() == "hi"

    async def test_a_refused_path_comes_back_as_a_result(self, registry, tmp_path):
        result = await registry.call("fs__read", {"path": str(tmp_path / "outside.txt")})
        assert result.ok is False
        assert "outside the allowed directories" in result.content

    async def test_a_missing_required_argument_is_reported(self, registry):
        assert (await registry.call("fs__read", {})).ok is False

    def test_every_tool_refuses_an_out_of_sandbox_path_up_front(self, registry, tmp_path):
        """The precheck lets the policy engine refuse without asking the user."""
        outside = str(tmp_path / "outside.txt")
        for name in registry.names():
            spec = registry.get(name)
            arguments = (
                {"source": outside, "destination": outside}
                if "source" in spec.input_schema["properties"] else {"path": outside}
            )
            assert spec.precheck(arguments), f"{name} has no path precheck"

    def test_the_precheck_passes_for_a_path_inside(self, registry, root):
        assert registry.get("fs__read").precheck({"path": str(root / "a.txt")}) is None

    def test_the_relocation_precheck_checks_both_ends(self, registry, root, tmp_path):
        spec = registry.get("fs__move")
        assert spec.precheck(
            {"source": str(root / "a"), "destination": str(tmp_path / "out")}
        )
        assert spec.precheck(
            {"source": str(tmp_path / "out"), "destination": str(root / "a")}
        )
        assert spec.precheck(
            {"source": str(root / "a"), "destination": str(root / "b")}
        ) is None
