"""Tests for local document search (jarvis/rag, tools/local/documents.py).

The thing a retrieval test must actually measure is retrieval: that asking
about a topic returns the passage about that topic, not merely that the code
runs. The ollama double embeds by word overlap, so "nearest" means something
here rather than being noise the assertions are fitted to.
"""

from pathlib import Path

import numpy as np
import pytest

from jarvis.rag.embeddings import Embedder, EmbeddingUnavailable
from jarvis.rag.indexer import Indexer, chunk_text
from jarvis.rag.store import DocumentStore, IndexMismatch
from jarvis.tools.local.documents import ORIGIN, build_tools


def vectors(*rows):
    return np.asarray(rows, dtype=np.float32)


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("a short note", size=1000, overlap=100) == ["a short note"]

    def test_empty_text_yields_nothing(self):
        assert chunk_text("   \n\n  ", size=1000, overlap=100) == []

    def test_paragraphs_are_the_first_seam(self):
        """Cutting on blank lines keeps each chunk about one thing."""
        text = "\n\n".join(["alpha " * 60, "beta " * 60])
        chunks = chunk_text(text, size=400, overlap=0)
        assert len(chunks) == 2
        assert "beta" not in chunks[0]

    def test_chunks_respect_the_budget(self):
        text = "\n\n".join(f"paragraph {i} " * 30 for i in range(10))
        for chunk in chunk_text(text, size=500, overlap=50):
            assert len(chunk) <= 500 + 50 + 2

    def test_an_over_long_paragraph_is_cut_on_sentences(self):
        """Cutting mid-sentence is what makes retrieved context read like
        nonsense, so it is the last resort, not the method."""
        text = " ".join(f"This is sentence number {i}." for i in range(60))
        chunks = chunk_text(text, size=300, overlap=0)
        assert len(chunks) > 1
        assert all(chunk.strip().endswith(".") for chunk in chunks)

    def test_a_sentence_longer_than_the_budget_is_still_cut(self):
        """A minified file or a table has no punctuation to cut on, and
        refusing to index it would be worse than cutting it."""
        chunks = chunk_text("x" * 5000, size=500, overlap=0)
        assert len(chunks) >= 10
        assert all(len(chunk) <= 500 for chunk in chunks)

    def test_overlap_repeats_the_tail(self):
        """An answer straddling a boundary has to be findable from either
        side."""
        text = "\n\n".join(f"paragraph number {i} with some words" for i in range(20))
        plain = chunk_text(text, size=200, overlap=0)
        overlapped = chunk_text(text, size=200, overlap=80)
        assert sum(len(c) for c in overlapped) > sum(len(c) for c in plain)

    def test_a_silly_overlap_does_not_loop_forever(self):
        chunks = chunk_text("word " * 500, size=100, overlap=999)
        assert chunks


class TestStore:
    def test_an_empty_index_returns_nothing(self, document_store):
        assert document_store.search(vectors([1, 0, 0])[0]) == []

    def test_a_stored_chunk_comes_back(self, document_store, tmp_path):
        document_store.replace(
            tmp_path / "a.md", 1.0, 10, ["hello"], vectors([1, 0, 0])
        )
        hits = document_store.search(np.asarray([1, 0, 0], dtype=np.float32))
        assert [hit.text for hit in hits] == ["hello"]

    def test_the_nearest_chunk_wins(self, document_store, tmp_path):
        document_store.replace(
            tmp_path / "a.md", 1.0, 10,
            ["about cats", "about boats", "about trains"],
            vectors([1, 0, 0], [0, 1, 0], [0, 0, 1]),
        )
        hits = document_store.search(np.asarray([0, 0.9, 0.1], dtype=np.float32))
        assert hits[0].text == "about boats"

    def test_results_are_ordered_best_first(self, document_store, tmp_path):
        document_store.replace(
            tmp_path / "a.md", 1.0, 10, ["near", "middle", "far"],
            vectors([1, 0, 0], [0.7, 0.7, 0], [0, 0, 1]),
        )
        hits = document_store.search(
            np.asarray([1, 0, 0], dtype=np.float32), limit=3
        )
        assert [hit.text for hit in hits] == ["near", "middle", "far"]
        assert hits[0].score > hits[1].score > hits[2].score

    def test_magnitude_does_not_decide_the_ranking(self, document_store, tmp_path):
        """Cosine, not dot product: a long chunk must not win for being long.

        The loud vector points partly the right way, which is what makes this
        discriminating - against a merely orthogonal one, a dot product would
        rank correctly too and the test would prove nothing.
        """
        document_store.replace(
            tmp_path / "a.md", 1.0, 10, ["right direction", "loud but vaguer"],
            vectors([1, 0, 0], [45, 45, 0]),
        )
        hits = document_store.search(np.asarray([1, 0, 0], dtype=np.float32))
        assert hits[0].text == "right direction"
        assert hits[0].score > hits[1].score

    def test_the_limit_is_honoured(self, document_store, tmp_path):
        document_store.replace(
            tmp_path / "a.md", 1.0, 10, [f"chunk {i}" for i in range(10)],
            vectors(*[[1, i / 10, 0] for i in range(10)]),
        )
        assert len(document_store.search(
            np.asarray([1, 0, 0], dtype=np.float32), limit=3
        )) == 3

    def test_asking_for_more_than_exists_is_fine(self, document_store, tmp_path):
        document_store.replace(tmp_path / "a.md", 1.0, 10, ["only"], vectors([1, 0]))
        assert len(document_store.search(
            np.asarray([1, 0], dtype=np.float32), limit=50
        )) == 1

    def test_hits_carry_their_source(self, document_store, tmp_path):
        path = tmp_path / "notes.md"
        document_store.replace(path, 1.0, 10, ["a", "b"], vectors([1, 0], [0, 1]))
        hit = document_store.search(np.asarray([0, 1], dtype=np.float32))[0]
        assert hit.path == str(path)
        assert hit.ordinal == 1

    def test_reindexing_replaces_rather_than_duplicates(
        self, document_store, tmp_path
    ):
        path = tmp_path / "a.md"
        document_store.replace(path, 1.0, 10, ["old"], vectors([1, 0]))
        document_store.replace(path, 2.0, 20, ["new"], vectors([1, 0]))
        assert document_store.counts() == (1, 1)
        assert document_store.search(
            np.asarray([1, 0], dtype=np.float32)
        )[0].text == "new"

    def test_forgetting_a_document_drops_its_chunks(self, document_store, tmp_path):
        path = tmp_path / "a.md"
        document_store.replace(path, 1.0, 10, ["x", "y"], vectors([1, 0], [0, 1]))
        assert document_store.forget(path) is True
        assert document_store.counts() == (0, 0)

    def test_forgetting_what_was_never_there_is_not_an_error(
        self, document_store, tmp_path
    ):
        assert document_store.forget(tmp_path / "absent.md") is False

    def test_an_unchanged_file_is_not_reindexed(self, document_store, tmp_path):
        path = tmp_path / "a.md"
        document_store.replace(path, 1.5, 10, ["x"], vectors([1, 0]))
        assert document_store.needs_reindex(path, 1.5, 10) is False
        assert document_store.needs_reindex(path, 1.5, 11) is True
        assert document_store.needs_reindex(path, 2.5, 10) is True

    def test_an_unseen_file_needs_indexing(self, document_store, tmp_path):
        assert document_store.needs_reindex(tmp_path / "new.md", 1.0, 1) is True

    def test_chunks_and_vectors_must_line_up(self, document_store, tmp_path):
        with pytest.raises(ValueError):
            document_store.replace(
                tmp_path / "a.md", 1.0, 10, ["one", "two"], vectors([1, 0])
            )


class TestModelConsistency:
    """Vectors from two models are not comparable, and the failure is silent:
    search keeps working and returns confident nonsense."""

    def test_the_model_is_recorded(self, document_store):
        document_store.claim("nomic-embed-text", 768)
        assert document_store.model() == ("nomic-embed-text", 768)

    def test_a_different_model_is_refused_at_index_time(self, document_store):
        document_store.claim("nomic-embed-text", 768)
        with pytest.raises(IndexMismatch):
            document_store.claim("mxbai-embed-large", 1024)

    def test_the_same_model_again_is_fine(self, document_store):
        document_store.claim("nomic-embed-text", 768)
        document_store.claim("nomic-embed-text", 768)

    def test_a_different_model_is_refused_at_search_time(self, document_store):
        document_store.claim("nomic-embed-text", 768)
        with pytest.raises(IndexMismatch) as raised:
            document_store.require("mxbai-embed-large")
        assert "re-index" in str(raised.value).lower()

    def test_an_empty_index_accepts_any_model(self, document_store):
        document_store.require("anything")

    def test_a_wrong_sized_query_is_refused(self, document_store, tmp_path):
        """The last line of defence, if the names ever agree and the vectors
        do not."""
        document_store.replace(tmp_path / "a.md", 1.0, 10, ["x"], vectors([1, 0, 0]))
        with pytest.raises(IndexMismatch):
            document_store.search(np.asarray([1, 0], dtype=np.float32))


class TestEmbedder:
    async def test_it_embeds_a_batch(self, fake_ollama):
        matrix = await Embedder(model="nomic-embed-text").embed(["one", "two"])
        assert matrix.shape == (2, fake_ollama.embed_dimensions)

    async def test_an_empty_batch_costs_no_request(self, fake_ollama):
        await Embedder().embed([])
        assert fake_ollama.embed_calls == []

    async def test_it_batches_long_inputs(self, fake_ollama):
        """One request per chunk would make indexing an archive unbearable."""
        await Embedder(batch_size=4).embed([f"text {i}" for i in range(10)])
        assert [len(call["input"]) for call in fake_ollama.embed_calls] == [4, 4, 2]

    async def test_the_configured_model_is_used(self, fake_ollama, settings):
        settings.rag_embed_model = "mxbai-embed-large"
        await Embedder().embed(["x"])
        assert fake_ollama.embed_calls[0]["model"] == "mxbai-embed-large"

    async def test_it_falls_back_like_a_chat_turn(self, fake_ollama, settings):
        """Indexing against an unreachable NAS should move to the laptop, not
        fail the run halfway through an archive."""
        settings.ollama_host = "http://nas:11434"
        settings.ollama_fallback_host = "http://localhost:11434"
        fake_ollama.host_models["http://nas:11434"] = ["llama3.1:70b"]
        fake_ollama.host_models["http://localhost:11434"] = ["llama3.2:3b"]
        fake_ollama.host_embed_errors["http://nas:11434"] = ConnectionError("down")

        matrix = await Embedder().embed(["x"])
        assert matrix.shape[0] == 1
        assert fake_ollama.embed_calls[-1]["host"] == "http://localhost:11434"

    async def test_no_provider_is_a_clear_refusal(self, fake_ollama):
        """The fix is a pull command, so the message has to name it."""
        fake_ollama.embed_error = ConnectionError("refused")
        with pytest.raises(EmbeddingUnavailable) as raised:
            await Embedder(model="nomic-embed-text").embed(["x"])
        assert "ollama pull nomic-embed-text" in str(raised.value)

    async def test_a_chat_model_asked_to_embed_is_caught(self, fake_ollama):
        """Pointing RAG_EMBED_MODEL at llama3.1 is an easy mistake and would
        otherwise index garbage."""
        fake_ollama.embedding_override = []
        with pytest.raises(EmbeddingUnavailable):
            await Embedder().embed(["x"])

    async def test_a_short_response_is_not_silently_accepted(self, fake_ollama):
        """Fewer vectors than inputs would misalign every chunk after it."""
        class Short:
            @staticmethod
            def get(key, default=None):
                return [[1.0, 0.0]] if key == "embeddings" else default

        async def short_embed(model="", input="", **kwargs):
            return Short()

        embedder = Embedder()
        provider = (await embedder.providers.candidates())[0]
        embedder.providers.client_for(provider).embed = short_embed

        with pytest.raises(EmbeddingUnavailable):
            await embedder.embed(["one", "two"])


class TestIndexer:
    def write(self, sandbox, name, text):
        path = sandbox / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    async def test_it_indexes_text_files(self, sandbox, document_store, fake_ollama):
        self.write(sandbox, "notes.md", "the plumber is called Dubois")
        summary = await Indexer(store=document_store).index([sandbox])
        assert summary["indexed"] == 1
        assert document_store.counts()[1] >= 1

    async def test_unknown_extensions_are_left_alone(
        self, sandbox, document_store, fake_ollama
    ):
        self.write(sandbox, "photo.jpg", "not really a photo")
        summary = await Indexer(store=document_store).index([sandbox])
        assert summary["indexed"] == 0

    async def test_the_sandbox_applies(self, sandbox, document_store, fake_ollama,
                                       tmp_path):
        """Indexing a file puts its contents one similarity match away from
        the prompt, so it obeys the same boundary as reading one."""
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "secret.md").write_text("private")

        summary = await Indexer(store=document_store).index([outside])
        assert summary["indexed"] == 0
        assert document_store.counts() == (0, 0)

    async def test_denied_patterns_apply_inside_an_allowed_directory(
        self, sandbox, document_store, fake_ollama, settings
    ):
        """DENIED_PATTERNS exists for the key sitting in a directory you did
        allow."""
        settings.denied_patterns = "*.key,.ssh"
        self.write(sandbox, ".ssh/notes.md", "a passphrase")
        self.write(sandbox, "fine.md", "ordinary notes")

        await Indexer(store=document_store).index([sandbox])
        assert all(".ssh" not in p for p in document_store.indexed_paths())

    async def test_unchanged_files_are_skipped_on_a_second_run(
        self, sandbox, document_store, fake_ollama
    ):
        """An indexer that re-embeds everything is one nobody runs twice."""
        self.write(sandbox, "notes.md", "some content here")
        indexer = Indexer(store=document_store)

        await indexer.index([sandbox])
        fake_ollama.embed_calls.clear()
        summary = await indexer.index([sandbox])

        assert summary["skipped"] == 1
        assert fake_ollama.embed_calls == []

    async def test_force_reindexes_anyway(self, sandbox, document_store, fake_ollama):
        self.write(sandbox, "notes.md", "some content here")
        indexer = Indexer(store=document_store)
        await indexer.index([sandbox])

        summary = await indexer.index([sandbox], force=True)
        assert summary["indexed"] == 1

    async def test_a_changed_file_is_reindexed(
        self, sandbox, document_store, fake_ollama
    ):
        path = self.write(sandbox, "notes.md", "first version")
        indexer = Indexer(store=document_store)
        await indexer.index([sandbox])

        path.write_text("a completely different second version")
        import os
        os.utime(path, (2_000_000_000, 2_000_000_000))

        summary = await indexer.index([sandbox])
        assert summary["indexed"] == 1

    async def test_a_deleted_file_leaves_the_index(
        self, sandbox, document_store, fake_ollama
    ):
        """A stale index is worse than none: it answers with text that is no
        longer anywhere on disk."""
        path = self.write(sandbox, "notes.md", "temporary content")
        indexer = Indexer(store=document_store)
        await indexer.index([sandbox])

        path.unlink()
        summary = await indexer.index([sandbox])

        assert summary["removed"] == 1
        assert document_store.counts() == (0, 0)

    async def test_a_document_outside_the_walked_roots_is_kept(
        self, sandbox, document_store, fake_ollama, settings
    ):
        """Not looked at is not the same as gone."""
        other = sandbox / "other"
        other.mkdir()
        self.write(sandbox, "kept.md", "content that stays")
        (other / "walked.md").write_text("content under the walked root")

        indexer = Indexer(store=document_store)
        await indexer.index([sandbox])
        await indexer.index([other])

        assert any("kept.md" in p for p in document_store.indexed_paths())

    async def test_binary_files_are_skipped_not_fatal(
        self, sandbox, document_store, fake_ollama
    ):
        (sandbox / "broken.md").write_bytes(b"\xff\xfe\x00 not utf-8")
        self.write(sandbox, "fine.md", "readable text")

        summary = await Indexer(store=document_store).index([sandbox])
        assert summary["indexed"] == 1

    async def test_oversized_files_are_skipped(
        self, sandbox, document_store, fake_ollama, settings
    ):
        settings.rag_max_file_bytes = 100
        self.write(sandbox, "huge.md", "x" * 500)
        assert (await Indexer(store=document_store).index([sandbox]))["indexed"] == 0

    async def test_an_empty_file_is_not_indexed(
        self, sandbox, document_store, fake_ollama
    ):
        self.write(sandbox, "empty.md", "   \n\n  ")
        assert (await Indexer(store=document_store).index([sandbox]))["indexed"] == 0

    async def test_the_model_is_claimed_on_the_index(
        self, sandbox, document_store, fake_ollama, settings
    ):
        settings.rag_embed_model = "nomic-embed-text"
        self.write(sandbox, "notes.md", "content")
        await Indexer(store=document_store).index([sandbox])
        assert document_store.model()[0] == "nomic-embed-text"


class TestSearchTool:
    def tool(self, store, **kwargs):
        return build_tools(store, Embedder(), **kwargs)[0]

    def indexed(self, store, sandbox, fake_ollama, **files):
        """Index some documents through the real pipeline."""
        import asyncio
        for name, text in files.items():
            (sandbox / f"{name}.md").write_text(text)
        asyncio.run(Indexer(store=store).index([sandbox]))
        return store

    def test_it_is_registered_with_a_usable_schema(self, document_store):
        spec = self.tool(document_store)
        assert spec.name == "docs__search"
        assert spec.input_schema["required"] == ["query"]

    def test_it_only_reads(self, document_store):
        from jarvis.tools.schema import Risk

        spec = self.tool(document_store)
        assert spec.risk is Risk.READ_ONLY
        assert spec.egress is False

    def test_an_empty_query_is_refused(self, document_store):
        result = self.tool(document_store).handler(query="  ")
        assert result.ok is False

    def test_an_empty_index_says_how_to_fill_it(self, document_store):
        result = self.tool(document_store).handler(query="anything")
        assert result.ok is False
        assert "jarvis-index" in result.content

    def test_it_finds_the_document_about_the_topic(
        self, document_store, sandbox, fake_ollama
    ):
        """The actual job: ask about a topic, get the passage about it first.

        Both documents come back when the limit allows it - that is what a
        limit of five over two documents means - so what is measured here is
        the ranking, not the exclusion.
        """
        self.indexed(
            document_store, sandbox, fake_ollama,
            plumbing="the plumber is called Dubois and his number is on the fridge",
            recipes="carbonara needs guanciale pecorino and eggs",
        )
        result = self.tool(document_store).handler(query="plumber Dubois")
        assert result.ok is True
        assert result.content.index("Dubois") < result.content.index("carbonara")

    def test_the_best_match_is_the_one_kept_when_only_one_fits(
        self, document_store, sandbox, fake_ollama
    ):
        self.indexed(
            document_store, sandbox, fake_ollama,
            plumbing="the plumber is called Dubois and his number is on the fridge",
            recipes="carbonara needs guanciale pecorino and eggs",
        )
        result = self.tool(document_store).handler(query="plumber Dubois", limit=1)
        assert "Dubois" in result.content
        assert "carbonara" not in result.content

    def test_results_name_their_source_file(
        self, document_store, sandbox, fake_ollama
    ):
        """So the model can cite it and the user can check the answer."""
        self.indexed(document_store, sandbox, fake_ollama,
                     notes="the plumber is called Dubois")
        result = self.tool(document_store).handler(query="plumber")
        assert "notes.md" in result.content

    def test_results_are_marked_untrusted(
        self, document_store, sandbox, fake_ollama
    ):
        """A similarity score chose this text, not a person: the index may
        hold a README from a cloned repository."""
        self.indexed(document_store, sandbox, fake_ollama, notes="some content")
        result = self.tool(document_store).handler(query="content")
        assert result.untrusted is True

    def test_every_result_shares_one_origin(
        self, document_store, sandbox, fake_ollama
    ):
        """So searching twice in a turn does not escalate twice - that is what
        the origin rule is for."""
        self.indexed(document_store, sandbox, fake_ollama, a="alpha", b="beta")
        first = self.tool(document_store).handler(query="alpha")
        second = self.tool(document_store).handler(query="beta")
        assert first.origin == second.origin == ORIGIN

    def test_the_limit_is_capped(self, document_store, sandbox, fake_ollama):
        """A model asking for 500 passages would fill the context window with
        the thing it was trying to avoid reading."""
        self.indexed(document_store, sandbox, fake_ollama,
                     **{f"note{i}": f"content number {i}" for i in range(25)})
        result = self.tool(document_store).handler(query="content", limit=500)
        assert result.content.count("--- ") <= 20

    def test_a_nonsense_limit_falls_back_to_the_default(
        self, document_store, sandbox, fake_ollama
    ):
        self.indexed(document_store, sandbox, fake_ollama, a="alpha content")
        assert self.tool(document_store).handler(query="alpha", limit="lots").ok

    def test_long_passages_are_truncated(
        self, document_store, sandbox, fake_ollama, settings
    ):
        settings.rag_chunk_size = 9000
        self.indexed(document_store, sandbox, fake_ollama, long="word " * 2000)
        result = self.tool(document_store).handler(query="word")
        assert "..." in result.content

    def test_an_unreachable_embedder_is_reported_not_raised(
        self, document_store, sandbox, fake_ollama
    ):
        """A tool that raises takes down the turn; one that reports lets the
        model say what went wrong."""
        self.indexed(document_store, sandbox, fake_ollama, notes="content")
        fake_ollama.embed_error = ConnectionError("refused")

        result = self.tool(document_store).handler(query="content")
        assert result.ok is False
        assert "ollama pull" in result.content

    def test_a_model_mismatch_is_reported_not_answered(
        self, document_store, sandbox, fake_ollama, settings
    ):
        """Answering from incomparable vectors would be confident nonsense."""
        self.indexed(document_store, sandbox, fake_ollama, notes="content")
        settings.rag_embed_model = "some-other-model"

        result = build_tools(document_store, Embedder())[0].handler(query="content")
        assert result.ok is False
        assert "not comparable" in result.content


class TestRegistration:
    """The tool is only offered once there is an index to search."""

    def test_no_index_means_no_tool(self, settings, tmp_path):
        from jarvis.tools.builtin import build_document_tools

        settings.data_dir = str(tmp_path / "empty")
        assert build_document_tools() == []

    def test_an_empty_index_means_no_tool(self, settings, tmp_path, fake_ollama):
        """Offering a tool that can only answer "nothing is indexed" spends a
        slot in the toolbox on nothing."""
        from jarvis.tools.builtin import build_document_tools

        settings.data_dir = str(tmp_path)
        DocumentStore(settings.documents_path)
        assert build_document_tools() == []

    def test_an_index_with_content_registers_the_tool(
        self, settings, tmp_path, sandbox, fake_ollama
    ):
        import asyncio

        from jarvis.tools.builtin import build_default_registry, build_document_tools

        settings.data_dir = str(tmp_path)
        (sandbox / "notes.md").write_text("indexed content")
        store = DocumentStore(settings.documents_path)
        asyncio.run(Indexer(store=store).index([sandbox]))

        assert [spec.name for spec in build_document_tools()] == ["docs__search"]
        assert "docs__search" in build_default_registry().names()

    def test_an_unreadable_index_costs_the_tool_not_the_session(
        self, settings, tmp_path
    ):
        from jarvis.tools.builtin import build_document_tools

        settings.data_dir = str(tmp_path)
        settings.documents_path.parent.mkdir(parents=True, exist_ok=True)
        settings.documents_path.write_text("this is not a database")
        assert build_document_tools() == []


class TestIndexerCommand:
    """`jarvis-index`, which is how the index actually gets built."""

    def run(self, argv, capsys):
        from jarvis.rag.indexer import main

        code = main(argv)
        return code, capsys.readouterr()

    def test_status_reports_an_empty_index(self, settings, tmp_path, capsys):
        settings.data_dir = str(tmp_path)
        code, output = self.run(["--status"], capsys)
        assert code == 0
        assert "Documents: 0" in output.out

    def test_status_reports_what_is_indexed(
        self, settings, tmp_path, sandbox, fake_ollama, capsys
    ):
        settings.data_dir = str(tmp_path)
        (sandbox / "notes.md").write_text("content to index")

        assert self.run([str(sandbox)], capsys)[0] == 0
        code, output = self.run(["--status"], capsys)
        assert "Documents: 1" in output.out
        assert settings.rag_embed_model in output.out

    def test_it_indexes_the_given_paths(
        self, settings, tmp_path, sandbox, fake_ollama, capsys
    ):
        settings.data_dir = str(tmp_path)
        (sandbox / "notes.md").write_text("the plumber is called Dubois")

        code, output = self.run([str(sandbox)], capsys)
        assert code == 0
        assert "1 indexed" in output.out

    def test_it_defaults_to_the_allowed_directories(
        self, settings, tmp_path, sandbox, fake_ollama, capsys
    ):
        """Indexing what Jarvis is allowed to read is the sensible default,
        and saves typing the same path twice."""
        settings.data_dir = str(tmp_path)
        (sandbox / "notes.md").write_text("content")

        code, output = self.run([], capsys)
        assert code == 0
        assert str(sandbox) in output.out

    def test_a_second_run_skips_unchanged_files(
        self, settings, tmp_path, sandbox, fake_ollama, capsys
    ):
        settings.data_dir = str(tmp_path)
        (sandbox / "notes.md").write_text("content")

        self.run([str(sandbox)], capsys)
        code, output = self.run([str(sandbox)], capsys)
        assert "1 unchanged" in output.out

    def test_force_is_passed_through(
        self, settings, tmp_path, sandbox, fake_ollama, capsys
    ):
        settings.data_dir = str(tmp_path)
        (sandbox / "notes.md").write_text("content")

        self.run([str(sandbox)], capsys)
        code, output = self.run([str(sandbox), "--force"], capsys)
        assert "1 indexed" in output.out

    def test_a_model_mismatch_exits_with_a_reason(
        self, settings, tmp_path, sandbox, fake_ollama, capsys
    ):
        """Mixing two embedding models silently would poison the index."""
        settings.data_dir = str(tmp_path)
        (sandbox / "notes.md").write_text("content")
        self.run([str(sandbox)], capsys)

        settings.rag_embed_model = "some-other-model"
        (sandbox / "second.md").write_text("more content")
        code, output = self.run([str(sandbox)], capsys)

        assert code == 2
        assert "Refused" in output.err

    def test_an_unreachable_server_exits_with_the_pull_command(
        self, settings, tmp_path, sandbox, fake_ollama, capsys
    ):
        settings.data_dir = str(tmp_path)
        (sandbox / "notes.md").write_text("content")
        fake_ollama.embed_error = ConnectionError("refused")

        code, output = self.run([str(sandbox)], capsys)
        assert code == 1
        assert "ollama pull" in output.err

    def test_an_interruption_keeps_what_was_indexed(
        self, settings, tmp_path, sandbox, fake_ollama, capsys, monkeypatch
    ):
        """Ctrl-C halfway through an archive should not throw away the hour
        that already ran."""
        import jarvis.rag.indexer as indexer_module

        settings.data_dir = str(tmp_path)

        async def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(indexer_module.Indexer, "index", interrupted)
        code, output = self.run([str(sandbox)], capsys)
        assert code == 130
        assert "kept" in output.err


class TestIndexerEdges:
    async def test_an_unreadable_file_is_skipped(
        self, sandbox, document_store, fake_ollama, monkeypatch
    ):
        """A permission error on one file must not end the run."""
        path = sandbox / "notes.md"
        path.write_text("content")

        original = Path.read_text

        def refuse(self, *args, **kwargs):
            if self == path:
                raise PermissionError("nope")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", refuse)
        assert (await Indexer(store=document_store).index([sandbox]))["indexed"] == 0

    async def test_a_file_the_walk_cannot_stat_is_skipped(
        self, sandbox, document_store, fake_ollama, monkeypatch
    ):
        """A stale mount point in the middle of an archive should cost that
        file, not the run."""
        path = sandbox / "notes.md"
        path.write_text("content")

        original = Path.stat

        def vanish(self, *args, **kwargs):
            if self == path:
                raise OSError("no such file")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", vanish)
        assert (await Indexer(store=document_store).index([sandbox]))["indexed"] == 0

    async def test_a_single_file_can_be_indexed(
        self, sandbox, document_store, fake_ollama
    ):
        """`jarvis-index ~/Documents/notes.md` should work, not just a
        directory."""
        path = sandbox / "notes.md"
        path.write_text("content to index")
        assert (await Indexer(store=document_store).index([path]))["indexed"] == 1


class TestRelevanceFloor:
    """Nearest-k always returns something. Without a floor, a question the
    documents say nothing about comes back with the five least-irrelevant
    passages - which the model then summarises as if they answered it."""

    def indexed(self, store, sandbox, **files):
        import asyncio
        for name, text in files.items():
            (sandbox / f"{name}.md").write_text(text)
        asyncio.run(Indexer(store=store).index([sandbox]))

    def test_weak_matches_are_dropped(self, document_store, sandbox, fake_ollama):
        self.indexed(document_store, sandbox, recipes="carbonara guanciale pecorino")
        tool = build_tools(
            document_store, Embedder(), min_similarity=0.9
        )[0]
        result = tool.handler(query="quarterly revenue forecast")
        assert result.ok is True
        assert "Nothing in the 1 indexed document(s) matched" in result.content

    def test_a_real_match_still_gets_through(
        self, document_store, sandbox, fake_ollama
    ):
        self.indexed(document_store, sandbox, recipes="carbonara guanciale pecorino")
        tool = build_tools(document_store, Embedder(), min_similarity=0.5)[0]
        assert "carbonara" in tool.handler(query="carbonara guanciale").content

    def test_the_floor_is_off_by_default(
        self, document_store, sandbox, fake_ollama
    ):
        """Each embedding model has its own baseline for unrelated text, so a
        wrong default would silently return nothing at all."""
        self.indexed(document_store, sandbox, recipes="carbonara guanciale")
        result = build_tools(document_store, Embedder())[0].handler(query="unrelated")
        assert result.untrusted is True

    def test_a_nothing_matched_answer_is_still_attributed(
        self, document_store, sandbox, fake_ollama
    ):
        """It came from the index, so the turn should know that even when the
        index had nothing useful."""
        self.indexed(document_store, sandbox, recipes="carbonara")
        tool = build_tools(document_store, Embedder(), min_similarity=0.99)[0]
        assert tool.handler(query="something else entirely").origin == ORIGIN

    def test_an_unexpected_failure_is_reported_not_raised(
        self, document_store, sandbox, fake_ollama, monkeypatch
    ):
        """A tool that raises takes down the turn."""
        self.indexed(document_store, sandbox, notes="content")
        tool = build_tools(document_store, Embedder())[0]

        def explode(*args, **kwargs):
            raise MemoryError("the index is too large")

        monkeypatch.setattr(document_store, "search", explode)
        result = tool.handler(query="content")
        assert result.ok is False
        assert "could not search" in result.content
