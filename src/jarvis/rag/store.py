"""Where indexed documents and their embeddings live.

SQLite, like the approvals and the conversations, and brute-force cosine
similarity over numpy - not a vector database. The arithmetic says why: a
person's notes are tens of thousands of chunks, and 20 000 x 768 float32 is
61 MB and a few milliseconds per query. Adding chromadb to beat that would
buy nothing and cost a dependency, a daemon and a second place for the data
to go stale.

The one trap worth guarding is the embedding model. Vectors from two
different models are not comparable, and the failure is silent: search keeps
working and simply returns nonsense. So the model and its dimension are
stored with the index, and a mismatch is refused rather than averaged over.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    mtime       REAL NOT NULL,
    size        INTEGER NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_by_document ON chunks (document_id, ordinal);
"""

#: Vectors are stored as float32: half the bytes of float64, and the
#: difference is far below what a cosine ranking can notice.
DTYPE = np.float32


@dataclass(frozen=True)
class Hit:
    """One chunk that matched, and how well."""

    path: str
    ordinal: int
    text: str
    score: float


class IndexMismatch(RuntimeError):
    """The index was built with a different embedding model.

    Not a warning: the vectors are incomparable, so searching anyway returns
    confident nonsense, which is worse than an error.
    """


class DocumentStore:
    """Indexed chunks, and the nearest-neighbour search over them."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple[str, int, str]] = []
        with closing(self._connect()) as db:
            db.executescript(SCHEMA)
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    # -- the model the index was built with --------------------------------- #

    def model(self) -> tuple[str | None, int | None]:
        """The embedding model and dimension this index holds, if any."""
        with closing(self._connect()) as db:
            rows = dict(db.execute("SELECT key, value FROM meta").fetchall())
        dimension = rows.get("dimension")
        return rows.get("model"), int(dimension) if dimension else None

    def claim(self, model: str, dimension: int) -> None:
        """Record the model, or refuse if the index already has another."""
        stored, stored_dimension = self.model()
        if stored is not None and (stored, stored_dimension) != (model, dimension):
            raise IndexMismatch(
                f"this index was built with '{stored}' ({stored_dimension} "
                f"dimensions) and cannot be mixed with '{model}' ({dimension}). "
                f"Re-index, or point RAG_EMBED_MODEL back at '{stored}'."
            )

        with closing(self._connect()) as db:
            db.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [("model", model), ("dimension", str(dimension))],
            )
            db.commit()

    def require(self, model: str) -> None:
        """Refuse to search an index built by a different model."""
        stored, _ = self.model()
        if stored is not None and stored != model:
            raise IndexMismatch(
                f"this index was built with '{stored}', but RAG_EMBED_MODEL is "
                f"'{model}'. The vectors are not comparable; re-index or change "
                f"the setting back."
            )

    # -- writing ------------------------------------------------------------ #

    def needs_reindex(self, path: Path, mtime: float, size: int) -> bool:
        """Whether this file has changed since it was last indexed.

        Re-embedding an unchanged archive on every run would make the indexer
        something nobody runs twice.
        """
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT mtime, size FROM documents WHERE path = ?", (str(path),)
            ).fetchone()
        if row is None:
            return True
        return abs(row[0] - mtime) > 1e-6 or row[1] != size

    def replace(
        self,
        path: Path,
        mtime: float,
        size: int,
        chunks: list[str],
        embeddings: np.ndarray,
    ) -> int:
        """Store one document's chunks, replacing whatever was there."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        vectors = np.asarray(embeddings, dtype=DTYPE)
        with closing(self._connect()) as db:
            db.execute("DELETE FROM documents WHERE path = ?", (str(path),))
            cursor = db.execute(
                "INSERT INTO documents (path, mtime, size, indexed_at) "
                "VALUES (?, ?, ?, ?)",
                (str(path), mtime, size, datetime.now(timezone.utc).isoformat()),
            )
            document_id = cursor.lastrowid
            db.executemany(
                "INSERT INTO chunks (document_id, ordinal, text, embedding) "
                "VALUES (?, ?, ?, ?)",
                [
                    (document_id, ordinal, text, vectors[ordinal].tobytes())
                    for ordinal, text in enumerate(chunks)
                ],
            )
            db.commit()

        self._matrix = None
        return len(chunks)

    def forget(self, path: Path) -> bool:
        """Drop a document. Used for files that have gone away."""
        with closing(self._connect()) as db:
            changed = db.execute(
                "DELETE FROM documents WHERE path = ?", (str(path),)
            ).rowcount
            db.execute(
                "DELETE FROM chunks WHERE document_id NOT IN "
                "(SELECT id FROM documents)"
            )
            db.commit()
        if changed:
            self._matrix = None
        return bool(changed)

    def indexed_paths(self) -> list[str]:
        with closing(self._connect()) as db:
            return [row[0] for row in db.execute("SELECT path FROM documents")]

    def counts(self) -> tuple[int, int]:
        """How many documents and chunks are indexed."""
        with closing(self._connect()) as db:
            documents = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return documents, chunks

    # -- reading ------------------------------------------------------------ #

    def _load(self) -> np.ndarray:
        """The whole index as one matrix, built once and kept."""
        if self._matrix is not None:
            return self._matrix

        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT d.path, c.ordinal, c.text, c.embedding "
                "FROM chunks c JOIN documents d ON d.id = c.document_id "
                "ORDER BY c.id"
            ).fetchall()

        if not rows:
            self._rows = []
            self._matrix = np.empty((0, 0), dtype=DTYPE)
            return self._matrix

        self._rows = [(row[0], row[1], row[2]) for row in rows]
        vectors = np.stack([np.frombuffer(row[3], dtype=DTYPE) for row in rows])
        # Normalised once here, so a query is a single matrix-vector product
        # rather than a division per row.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._matrix = vectors / np.maximum(norms, 1e-12)
        logger.debug(f"Loaded {len(self._rows)} chunks for search")
        return self._matrix

    def search(self, embedding: np.ndarray, limit: int = 5) -> list[Hit]:
        """The chunks closest to one query vector, best first."""
        matrix = self._load()
        if matrix.size == 0:
            return []

        query = np.asarray(embedding, dtype=DTYPE).ravel()
        if query.shape[0] != matrix.shape[1]:
            raise IndexMismatch(
                f"the query has {query.shape[0]} dimensions and the index has "
                f"{matrix.shape[1]}; they were built by different models"
            )

        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = matrix @ query

        limit = max(1, min(limit, len(scores)))
        # argpartition finds the top k without sorting the rest, which matters
        # once an index is large and does not hurt when it is not.
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]

        return [
            Hit(
                path=self._rows[index][0],
                ordinal=self._rows[index][1],
                text=self._rows[index][2],
                score=float(scores[index]),
            )
            for index in top
        ]
