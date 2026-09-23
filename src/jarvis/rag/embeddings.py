"""Turning text into vectors, without adding a machine-learning stack.

`FEATURES_IDEA.md` proposed sentence-transformers. That means torch: a
multi-gigabyte install, on a machine that is already running a language model
and a speech recogniser, to do something the language model server does
natively. Ollama serves embedding models (`ollama pull nomic-embed-text`),
which means zero new Python dependencies and - because this goes through the
same provider pool as everything else - embeddings computed on the NAS when
the NAS is there.

Queries and documents go through the same code path on purpose. Embedding a
query with one model and the documents with another is the classic way to get
a search that returns confident nonsense; the store refuses the mismatch, and
this keeps them from diverging in the first place.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
from loguru import logger

from jarvis.config import settings
from jarvis.llm_providers import ProviderPool
from jarvis.rag.store import DTYPE


class EmbeddingUnavailable(RuntimeError):
    """No provider could embed - unreachable, or the model is not pulled."""


class Embedder:
    """Embeds text through whichever Ollama provider is answering."""

    def __init__(
        self,
        model: str | None = None,
        providers: ProviderPool | None = None,
        batch_size: int | None = None,
    ):
        self.model = model or settings.rag_embed_model
        self.providers = providers if providers is not None else ProviderPool()
        self.batch_size = batch_size or settings.rag_batch_size

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch, as rows of a matrix.

        Providers are tried in order, exactly as a chat turn is: an embedding
        against an unreachable NAS should fall back to the laptop rather than
        fail the indexing run.
        """
        if not texts:
            return np.empty((0, 0), dtype=DTYPE)

        last_error: Exception | None = None
        for provider in await self.providers.candidates():
            try:
                vectors = await self._embed_with(provider, texts)
            except Exception as e:
                logger.warning(f"Embedding failed on {provider.describe()}: {e}")
                self.providers.report_failure(provider, e)
                last_error = e
                continue

            self.providers.report_success(provider)
            return vectors

        raise EmbeddingUnavailable(
            f"no provider could run '{self.model}': {last_error}. "
            f"Pull it with: ollama pull {self.model}"
        )

    async def _embed_with(self, provider, texts: Sequence[str]) -> np.ndarray:
        client = self.providers.client_for(provider)
        rows: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            response = await client.embed(model=self.model, input=batch)
            vectors = (
                response.get("embeddings") if hasattr(response, "get")
                else getattr(response, "embeddings", None)
            )
            if not vectors or len(vectors) != len(batch):
                raise EmbeddingUnavailable(
                    f"'{self.model}' returned {len(vectors or [])} vectors for "
                    f"{len(batch)} inputs"
                )
            rows.extend([list(vector) for vector in vectors])

        matrix = np.asarray(rows, dtype=DTYPE)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise EmbeddingUnavailable(
                f"'{self.model}' did not return usable vectors - is it an "
                f"embedding model?"
            )
        return matrix

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed a single query."""
        return (await self.embed([text]))[0]

    def embed_one_sync(self, text: str) -> np.ndarray:
        """Embed a query from synchronous code.

        The tool handlers are synchronous - the registry calls them in a
        worker thread - so a search cannot simply await.
        """
        return asyncio.run(self.embed_one(text))
