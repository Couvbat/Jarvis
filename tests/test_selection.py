"""Tests for tool selection and text helpers."""

import pytest

from text_utils import expand, normalise, strip_accents, tokenise
from tools.registry import ToolRegistry
from tools.schema import Risk, ToolResult, ToolSpec
from tools.selection import ToolSelector, estimate_schema_tokens


def make_spec(name, description="does something", properties=None):
    return ToolSpec(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties or {}},
        handler=lambda **kwargs: ToolResult("ok"),
        risk=Risk.READ_ONLY,
    )


def registry_of(*specs):
    instance = ToolRegistry()
    instance.register_all(specs)
    return instance


class TestTextHelpers:
    @pytest.mark.parametrize("raw,expected", [
        ("Crée", "cree"),
        ("Arrête-toi !", "arrete toi"),
        ("  spaced   out  ", "spaced out"),
        ("café.txt", "cafe txt"),
    ])
    def test_normalise(self, raw, expected):
        assert normalise(raw) == expected

    def test_strip_accents_keeps_case(self):
        assert strip_accents("Crée") == "Cree"

    def test_identifiers_split_into_parts(self):
        """Otherwise a tool name would only ever match itself."""
        assert tokenise("fs__read_file") == ["fs", "read", "file"]

    def test_camel_case_splits(self):
        assert tokenise("listNotes readNote") == ["list", "notes", "read", "note"]

    def test_accented_words_survive_intact(self):
        """Stripping accents after the split turned "Crée" into "cr" + "e"."""
        assert "cree" in tokenise("Crée un fichier")

    def test_stopwords_are_dropped(self):
        assert tokenise("the file in my documents") == ["file", "documents"]

    def test_stopwords_can_be_kept(self):
        assert "the" in tokenise("the file", keep_stopwords=True)

    def test_single_letters_are_dropped(self):
        assert tokenise("a b file") == ["file"]

    def test_french_terms_expand_to_english(self):
        """Tool descriptions are English; the speaker may not be."""
        assert "file" in expand(["fichier"])
        assert "delete" in expand(["supprime"])

    def test_english_synonyms_expand_too(self):
        assert "launch" in expand(["open"])

    def test_the_original_terms_are_kept(self):
        assert "fichier" in expand(["fichier"])

    def test_unknown_terms_pass_through(self):
        assert expand(["marie"]) == ["marie"]


class TestBelowTheThreshold:
    def test_everything_is_offered(self):
        """Selection is a no-op for a small tool set: offering a slightly
        crowded toolbox beats offering a confidently wrong shortlist."""
        registry = registry_of(make_spec("fs__read"), make_spec("fs__write"))
        assert ToolSelector(threshold=5).select(registry, "anything") is None

    def test_the_threshold_is_inclusive(self):
        registry = registry_of(*(make_spec(f"a__{i}") for i in range(5)))
        assert ToolSelector(threshold=5).select(registry, "x") is None

    def test_one_more_tool_turns_selection_on(self):
        registry = registry_of(*(make_spec(f"a__{i}") for i in range(6)))
        assert ToolSelector(threshold=5, top_k=3).select(registry, "x") is not None


class TestRanking:
    @pytest.fixture
    def registry(self):
        return registry_of(
            make_spec("fs__read", "Read the contents of a text file"),
            make_spec("fs__write", "Write a file: create, overwrite or append"),
            make_spec("fs__delete", "Delete a file or a directory"),
            make_spec("web__fetch", "Fetch a web page and return its text"),
            make_spec("app__launch", "Launch an application"),
            make_spec("mail__send", "Send an email message to someone"),
        )

    def selector(self):
        return ToolSelector(threshold=1, top_k=3)

    def test_the_obvious_tool_comes_first(self, registry):
        chosen = self.selector().select(registry, "read me that file")
        assert chosen[0] == "fs__read"

    def test_french_reaches_english_descriptions(self, registry):
        chosen = self.selector().select(registry, "supprime le fichier")
        assert "fs__delete" in chosen

    def test_breadth_of_match_counts(self, registry):
        """One strong name hit should not beat two description hits."""
        chosen = self.selector().select(registry, "create a new file")
        assert "fs__write" in chosen

    def test_the_shortlist_is_capped(self, registry):
        assert len(self.selector().select(registry, "file")) == 3

    def test_an_unrelated_request_still_gets_tools(self, registry):
        """Never hand the model an empty toolbox because nothing matched."""
        chosen = self.selector().select(registry, "zzzz qqqq")
        assert len(chosen) == 3

    def test_parameter_names_contribute(self):
        registry = registry_of(
            *(make_spec(f"pad__{i}") for i in range(6)),
            make_spec("x__one", "does a thing", {"url": {"type": "string"}}),
        )
        chosen = ToolSelector(threshold=1, top_k=2).select(registry, "url")
        assert "x__one" in chosen


class TestConversationContext:
    @pytest.fixture
    def registry(self):
        return registry_of(
            make_spec("fs__read", "Read the contents of a text file"),
            make_spec("fs__delete", "Delete a file or a directory"),
            make_spec("mail__send", "Send an email message to someone"),
            make_spec("web__fetch", "Fetch a web page"),
            make_spec("app__launch", "Launch an application"),
            make_spec("git__log", "Show recent commits"),
        )

    def test_a_follow_up_uses_what_came_before(self, registry):
        """"and delete it" names nothing on its own."""
        selector = ToolSelector(threshold=1, top_k=2)
        chosen = selector.select(
            registry, "et supprime-le", ["lis le fichier brouillon"]
        )
        assert "fs__delete" in chosen

    def test_recently_used_tools_stay_available(self, registry):
        selector = ToolSelector(threshold=1, top_k=2)
        selector.note_used("git__log")
        assert "git__log" in selector.select(registry, "lis le fichier")

    def test_only_the_last_few_used_are_kept(self, registry):
        selector = ToolSelector(threshold=1, top_k=6, recent_size=2)
        for name in ("fs__read", "mail__send", "web__fetch"):
            selector.note_used(name)
        assert selector._recent == ["mail__send", "web__fetch"]

    def test_a_tool_used_twice_is_not_duplicated(self, registry):
        selector = ToolSelector(threshold=1, top_k=4)
        selector.note_used("git__log")
        selector.note_used("git__log")
        chosen = selector.select(registry, "x")
        assert chosen.count("git__log") == 1

    def test_a_tool_that_went_away_is_not_offered(self, registry):
        """An MCP server can disconnect between turns."""
        selector = ToolSelector(threshold=1, top_k=3)
        selector.note_used("gone__tool")
        assert "gone__tool" not in selector.select(registry, "x")

    def test_reset_forgets_recent_use(self, registry):
        selector = ToolSelector(threshold=1, top_k=1)
        selector.note_used("git__log")
        selector.reset()
        assert "git__log" not in selector.select(registry, "lis le fichier")


class TestIndexing:
    def test_the_index_follows_a_changing_tool_set(self):
        """MCP servers come and go mid-session."""
        registry = registry_of(*(make_spec(f"a__{i}") for i in range(6)))
        selector = ToolSelector(threshold=1, top_k=3)
        selector.select(registry, "x")

        registry.register(make_spec("mail__send", "Send an email message"))
        chosen = selector.select(registry, "send an email")
        assert "mail__send" in chosen

    def test_a_removed_namespace_drops_out(self):
        registry = registry_of(
            *(make_spec(f"a__{i}") for i in range(6)),
            make_spec("gone__tool", "Send an email message"),
        )
        selector = ToolSelector(threshold=1, top_k=3)
        selector.select(registry, "send an email")
        registry.unregister_namespace("gone")
        assert "gone__tool" not in selector.select(registry, "send an email")


class TestTokenEstimate:
    def test_more_schemas_cost_more(self):
        one = [{"function": {"name": "a", "description": "x" * 100}}]
        assert estimate_schema_tokens(one * 3) > estimate_schema_tokens(one)

    def test_nothing_costs_nothing(self):
        assert estimate_schema_tokens([]) == 0
