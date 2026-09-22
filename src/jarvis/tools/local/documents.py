"""Searching the indexed documents.

Retrieval is a tool the model calls, not context injected into every turn.
Injecting it would spend the context window on "what time is it" and pull
unrelated notes into every answer; a tool means the model asks when the
question is about the user's documents, and the existing selection layer
already decides when to offer it.

Results are marked untrusted, which `fs__read` is not, and the difference is
worth stating. With `fs__read` a person named the file. Here a similarity
score chose it, and the index may hold a README from a cloned repository or
a downloaded document that says "ignore your instructions". Because `origin`
is the index rather than each file, searching again does not re-escalate -
that is what the origin rule in policy/taint.py exists for - but a write
after a search asks once, at the keyboard.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from jarvis.rag.embeddings import Embedder, EmbeddingUnavailable
from jarvis.rag.store import DocumentStore, IndexMismatch
from jarvis.tools.schema import Risk, ToolResult, ToolSpec, namespaced

NAMESPACE = "docs"

#: Everything retrieved shares this origin: re-reading the index is not the
#: exfiltration shape, so it must not escalate a second time.
ORIGIN = "local document index"

#: Characters of each chunk handed to the model. A retrieved chunk that fills
#: the context window defeats the point of retrieving a few of them.
MAX_CHUNK_CHARS = 1500


class DocumentTools:
    """The search side of the document index."""

    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder,
        default_limit: int = 5,
        min_similarity: float = 0.0,
    ):
        self.store = store
        self.embedder = embedder
        self.default_limit = default_limit
        self.min_similarity = min_similarity

    def search(self, query: str = "", limit: Any = None, **_: Any) -> ToolResult:
        """Find the passages closest to a question."""
        text = (query or "").strip()
        if not text:
            return ToolResult.error("a search needs a query")

        try:
            count = int(limit) if limit else self.default_limit
        except (TypeError, ValueError):
            count = self.default_limit
        count = max(1, min(count, 20))

        documents, chunks = self.store.counts()
        if not chunks:
            return ToolResult.error(
                "no documents are indexed yet; the user can index some with: "
                "jarvis-index ~/Documents"
            )

        try:
            self.store.require(self.embedder.model)
            vector = self.embedder.embed_one_sync(text)
            hits = self.store.search(vector, limit=count)
        except IndexMismatch as e:
            return ToolResult.error(str(e))
        except EmbeddingUnavailable as e:
            return ToolResult.error(str(e))
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return ToolResult.error(f"could not search the documents: {e}")

        # Nearest-k always returns something, so a question the documents say
        # nothing about would otherwise come back with the five
        # least-irrelevant passages - and be summarised as if they answered it.
        hits = [hit for hit in hits if hit.score >= self.min_similarity]
        if not hits:
            return ToolResult(
                f"Nothing in the {documents} indexed document(s) matched "
                f"'{text}'.",
                origin=ORIGIN,
            )

        blocks = []
        for hit in hits:
            body = hit.text.strip()
            if len(body) > MAX_CHUNK_CHARS:
                body = body[:MAX_CHUNK_CHARS].rstrip() + "..."
            # The source is on its own line so the model can cite it, and so
            # the user can check where an answer came from.
            blocks.append(
                f"--- {hit.path} (passage {hit.ordinal + 1}, "
                f"similarity {hit.score:.2f})\n{body}"
            )

        logger.info(f"Document search '{text}': {len(hits)} passage(s)")
        return ToolResult("\n\n".join(blocks), untrusted=True, origin=ORIGIN)


def build_tools(
    store: DocumentStore,
    embedder: Embedder,
    default_limit: int = 5,
    min_similarity: float = 0.0,
) -> list[ToolSpec]:
    """Build the document tools."""
    tools = DocumentTools(store, embedder, default_limit, min_similarity)
    return [
        ToolSpec(
            name=namespaced(NAMESPACE, "search"),
            description=(
                "Search the user's own indexed documents and notes for "
                "passages about a topic, and return them with their file "
                "paths. Use this for questions about what the user has "
                "written, saved or read - meeting notes, documentation, "
                "personal files - rather than guessing or asking them to "
                "find the file themselves."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, in natural language",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many passages to return (default 5)",
                    },
                },
                "required": ["query"],
            },
            handler=tools.search,
            # Reads the user's own disk, and only what they chose to index.
            risk=Risk.READ_ONLY,
            scope_for=lambda arguments: "the document index",
        ),
    ]
