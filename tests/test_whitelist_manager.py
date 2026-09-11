"""Tests for the persistent approval store (whitelist_manager.py)."""

import json

import pytest

from whitelist_manager import WhitelistManager

CATEGORIES = ("file_operations", "applications", "web_urls")


@pytest.fixture
def store(tmp_path):
    return WhitelistManager(str(tmp_path / "command_whitelist.json"))


class TestLoading:
    def test_missing_file_yields_empty_categories(self, store):
        for category in CATEGORIES:
            assert store.get_whitelist(category) == set()

    def test_missing_file_is_not_created_eagerly(self, tmp_path):
        path = tmp_path / "command_whitelist.json"
        WhitelistManager(str(path))
        assert not path.exists()

    def test_existing_file_is_loaded_as_sets(self, tmp_path):
        path = tmp_path / "command_whitelist.json"
        path.write_text(json.dumps({
            "file_operations": ["create_file:/tmp"],
            "applications": ["firefox"],
            "web_urls": ["example.com"],
        }))
        store = WhitelistManager(str(path))
        assert store.get_whitelist("applications") == {"firefox"}
        assert isinstance(store.get_whitelist("web_urls"), set)

    def test_corrupt_file_falls_back_to_empty(self, tmp_path):
        path = tmp_path / "command_whitelist.json"
        path.write_text("{ not json at all")
        store = WhitelistManager(str(path))
        assert store.get_whitelist("file_operations") == set()
        assert set(store.whitelist) == set(CATEGORIES)

    def test_partial_file_loses_default_categories(self, tmp_path):
        """A file with only one key leaves the other categories absent.

        ``get_whitelist`` and ``is_whitelisted`` both tolerate it, but
        ``whitelist`` no longer has the documented shape.
        """
        path = tmp_path / "command_whitelist.json"
        path.write_text(json.dumps({"applications": ["firefox"]}))
        store = WhitelistManager(str(path))
        assert set(store.whitelist) == {"applications"}
        assert store.get_whitelist("web_urls") == set()
        assert store.is_whitelisted("web_urls", "example.com") is False


class TestMutation:
    def test_add_then_query(self, store):
        store.add_to_whitelist("applications", "firefox")
        assert store.is_whitelisted("applications", "firefox") is True
        assert store.is_whitelisted("applications", "chromium") is False

    def test_add_is_idempotent(self, store):
        store.add_to_whitelist("web_urls", "example.com")
        store.add_to_whitelist("web_urls", "example.com")
        assert store.get_whitelist("web_urls") == {"example.com"}

    def test_add_creates_unknown_category(self, store):
        store.add_to_whitelist("brand_new", "thing")
        assert store.is_whitelisted("brand_new", "thing") is True

    def test_remove(self, store):
        store.add_to_whitelist("applications", "firefox")
        store.remove_from_whitelist("applications", "firefox")
        assert store.is_whitelisted("applications", "firefox") is False

    def test_remove_unknown_item_is_a_noop(self, store):
        store.remove_from_whitelist("applications", "never-added")
        store.remove_from_whitelist("no_such_category", "x")

    def test_lookup_of_unknown_category_is_false(self, store):
        assert store.is_whitelisted("no_such_category", "x") is False

    def test_get_whitelist_returns_a_copy(self, store):
        store.add_to_whitelist("applications", "firefox")
        snapshot = store.get_whitelist("applications")
        snapshot.add("mutated")
        assert store.is_whitelisted("applications", "mutated") is False


class TestPersistence:
    def test_add_writes_to_disk(self, tmp_path):
        path = tmp_path / "command_whitelist.json"
        WhitelistManager(str(path)).add_to_whitelist("applications", "firefox")
        assert json.loads(path.read_text()) == {
            "file_operations": [],
            "applications": ["firefox"],
            "web_urls": [],
        }

    def test_round_trip_across_instances(self, tmp_path):
        path = tmp_path / "command_whitelist.json"
        first = WhitelistManager(str(path))
        first.add_to_whitelist("file_operations", "create_file:/tmp")
        second = WhitelistManager(str(path))
        assert second.is_whitelisted("file_operations", "create_file:/tmp") is True

    def test_remove_is_persisted(self, tmp_path):
        path = tmp_path / "command_whitelist.json"
        first = WhitelistManager(str(path))
        first.add_to_whitelist("applications", "firefox")
        first.remove_from_whitelist("applications", "firefox")
        assert WhitelistManager(str(path)).get_whitelist("applications") == set()

    def test_unwritable_path_does_not_raise(self, tmp_path):
        """Persistence failures are swallowed - the in-memory set still updates."""
        directory = tmp_path / "not-a-file"
        directory.mkdir()
        store = WhitelistManager(str(directory))
        store.add_to_whitelist("applications", "firefox")
        assert store.is_whitelisted("applications", "firefox") is True

    def test_default_file_name(self):
        assert WhitelistManager().whitelist_file.name == "command_whitelist.json"
