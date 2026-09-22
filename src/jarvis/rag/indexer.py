"""Walking a directory, cutting documents up, and storing the vectors.

Chunking is on blank lines first, then sentences, then - only if a single
sentence is longer than the budget - on characters. Cutting mid-sentence is
what makes retrieved context read like nonsense, so it is the last resort
rather than the method.

Indexing is incremental by mtime and size: an unchanged archive is skipped.
An indexer that re-embeds everything on every run is one nobody runs twice,
and a stale index is worse than none.

The path sandbox applies here too. `jarvis-index ~/.ssh` is refused, and so
is any file matching DENIED_PATTERNS, because indexing a file is putting its
contents one similarity match away from the model's prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from jarvis.config import settings
from jarvis.policy.paths import PathPolicy
from jarvis.rag.embeddings import Embedder, EmbeddingUnavailable
from jarvis.rag.store import DocumentStore, IndexMismatch

#: Paragraph break: one or more blank lines.
_PARAGRAPH = re.compile(r"\n\s*\n")
#: Sentence end followed by whitespace. Crude, and good enough for splitting
#: an over-long paragraph - it is not parsing the text, only cutting it.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Cut text into overlapping pieces of roughly `size` characters."""
    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))

    pieces: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if paragraph:
            pieces.extend(_split_long(paragraph, size))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= size or not current:
            current = candidate
            continue
        chunks.append(current)
        # Carry the tail of the last chunk into the next one, so an answer
        # that straddles a boundary is still found from either side.
        current = (current[-overlap:] + "\n\n" + piece) if overlap else piece

    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, size: int) -> list[str]:
    """Break one over-long paragraph, on sentences where possible."""
    if len(paragraph) <= size:
        return [paragraph]

    parts: list[str] = []
    for sentence in _SENTENCE.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= size:
            parts.append(sentence)
            continue
        # A single sentence longer than the budget - a minified file, a table,
        # a wall of text with no punctuation. Nothing left but to cut it.
        parts.extend(
            sentence[start:start + size] for start in range(0, len(sentence), size)
        )
    return parts


class Indexer:
    """Builds and refreshes the document index."""

    def __init__(
        self,
        store: DocumentStore | None = None,
        embedder: Embedder | None = None,
        policy: PathPolicy | None = None,
    ):
        self.store = store if store is not None else DocumentStore(
            settings.documents_path
        )
        self.embedder = embedder if embedder is not None else Embedder()
        self.policy = policy if policy is not None else PathPolicy.from_settings()
        self.extensions = {
            extension if extension.startswith(".") else f".{extension}"
            for extension in settings.rag_extensions_list
        }

    # -- what to index ------------------------------------------------------ #

    def candidates(self, root: Path) -> Iterator[Path]:
        """Every file under `root` worth indexing, sandbox applied."""
        verdict = self.policy.check(root)
        if not verdict:
            logger.error(f"Refusing to index {root}: {verdict.reason}")
            return

        target = verdict.path
        files = [target] if target.is_file() else sorted(target.rglob("*"))

        for path in files:
            # One unreadable file - a stale mount point, a permission the
            # walk could see but the stat cannot - skips that file. A run
            # over an archive must not end on the first of them.
            try:
                if not path.is_file() or path.suffix.lower() not in self.extensions:
                    continue
                oversized = path.stat().st_size > settings.rag_max_file_bytes
            except OSError as e:
                logger.debug(f"Skipping {path}: {e}")
                continue

            if oversized:
                logger.debug(f"Skipping {path}: larger than the size limit")
                continue

            # Checked per file, not just per root: DENIED_PATTERNS exists to
            # catch the id_rsa sitting inside a directory you did allow.
            file_verdict = self.policy.check(path)
            if not file_verdict:
                logger.debug(f"Skipping {path}: {file_verdict.reason}")
                continue

            yield path

    @staticmethod
    def read(path: Path) -> str | None:
        """A file's text, or None when it is not text at all."""
        try:
            return path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            logger.debug(f"Skipping {path}: not UTF-8 text")
            return None
        except OSError as e:
            logger.warning(f"Could not read {path}: {e}")
            return None

    # -- indexing ----------------------------------------------------------- #

    async def index(self, roots: list[str | Path], force: bool = False) -> dict:
        """Index every root, and return what happened."""
        summary = {"indexed": 0, "skipped": 0, "chunks": 0, "removed": 0}
        seen: set[str] = set()

        for root in roots:
            for path in self.candidates(Path(root)):
                seen.add(str(path))
                stat = path.stat()
                if not force and not self.store.needs_reindex(
                    path, stat.st_mtime, stat.st_size
                ):
                    summary["skipped"] += 1
                    continue

                text = self.read(path)
                if text is None or not text.strip():
                    continue

                chunks = chunk_text(
                    text, settings.rag_chunk_size, settings.rag_chunk_overlap
                )
                if not chunks:
                    continue

                vectors = await self.embedder.embed(chunks)
                self.store.claim(self.embedder.model, int(vectors.shape[1]))
                self.store.replace(
                    path, stat.st_mtime, stat.st_size, chunks, vectors
                )
                summary["indexed"] += 1
                summary["chunks"] += len(chunks)
                logger.info(f"Indexed {path} ({len(chunks)} chunks)")

        summary["removed"] = self._forget_missing(seen, roots)
        return summary

    def _forget_missing(self, seen: set[str], roots: list[str | Path]) -> int:
        """Drop indexed files that have since been deleted.

        Only under the roots just walked: a document indexed from elsewhere
        has not been shown to be gone, merely not looked at.
        """
        resolved_roots = []
        for root in roots:
            verdict = self.policy.check(Path(root))
            if verdict:
                resolved_roots.append(verdict.path)

        removed = 0
        for indexed in self.store.indexed_paths():
            if indexed in seen:
                continue
            path = Path(indexed)
            if not any(path == root or root in path.parents for root in resolved_roots):
                continue
            if not path.exists() and self.store.forget(path):
                logger.info(f"Removed {path} from the index (file is gone)")
                removed += 1
        return removed


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jarvis-index",
        description=(
            "Index local documents so Jarvis can search them. Only paths "
            "inside ALLOWED_DIRECTORIES are indexed: indexing a file puts its "
            "contents one similarity match away from the model's prompt."
        ),
    )
    parser.add_argument(
        "paths", nargs="*",
        help="files or directories to index (default: ALLOWED_DIRECTORIES)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-embed even files that have not changed",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="show what is indexed and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """The `jarvis-index` command."""
    arguments = parse_arguments(argv)
    logger.remove()
    logger.add(sys.stderr, format="{message}", level=settings.log_level)

    store = DocumentStore(settings.documents_path)

    if arguments.status:
        documents, chunks = store.counts()
        model, dimension = store.model()
        print(f"Index:     {settings.documents_path}")
        print(f"Documents: {documents}")
        print(f"Chunks:    {chunks}")
        print(f"Model:     {model or 'none yet'} ({dimension or '-'} dimensions)")
        return 0

    roots = arguments.paths or [str(d) for d in settings.allowed_dirs_list]
    print(f"Indexing: {', '.join(str(r) for r in roots)}")

    try:
        summary = asyncio.run(Indexer(store=store).index(roots, force=arguments.force))
    except IndexMismatch as e:
        print(f"\nRefused: {e}", file=sys.stderr)
        return 2
    except EmbeddingUnavailable as e:
        print(f"\nCould not embed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; what was indexed so far is kept.", file=sys.stderr)
        return 130

    print(
        f"\n{summary['indexed']} indexed ({summary['chunks']} chunks), "
        f"{summary['skipped']} unchanged, {summary['removed']} removed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
