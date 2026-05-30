"""The `MemoryStore` interface -- the single seam between the engine and storage.

The MVP implements this over SQLite (sqlite-vec + FTS5). Phase 4 implements the
exact same interface over Postgres (pgvector + FTS). `test_store_conformance.py`
runs identically against both, which is what makes the SQLite->Postgres
graduation a swap rather than a rewrite.

Methods return domain models from `models.py` so callers stay storage-agnostic.
Candidate methods return `(id, score)` pairs in rank order for the hybrid fuser.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from knowledge_mcp.models import Edge, Memory, MemoryType

# A retrieval candidate: (memory_id, score) where higher score == more relevant.
Candidate = tuple[str, float]


class MemoryStore(ABC):
    @abstractmethod
    def init_schema(self) -> None:
        """Create tables/indices if absent. Idempotent."""

    # --- writes ---
    @abstractmethod
    def add_memory(self, memory: Memory, embedding: list[float]) -> str:
        """Persist a memory and index it (vector + lexical). Returns its id."""

    @abstractmethod
    def add_edge(self, edge: Edge) -> str:
        """Persist a bitemporal edge. Returns its id."""

    @abstractmethod
    def close_edge(
        self,
        edge_id: str,
        valid_to: datetime | None = None,
        expired_at: datetime | None = None,
    ) -> None:
        """Bitemporally close an edge (never deletes any row).

        Two distinct, composable closures:
          * `valid_to`   -- the relation stopped being true *in the world* (valid-time).
                            The row is still *believed*, so it remains visible to
                            currently-believed queries and to point-in-time queries
                            for instants inside its old validity window.
          * `expired_at` -- the row was *retracted/superseded* (transaction-time): we no
                            longer believe it (e.g. a correction). It drops out of all
                            currently-believed reads but is retained for audit.

        The canonical "project focus A -> B" shift is a *valid-time* change: close edge
        A with `valid_to` only, then add edge B. A query `as_of` an instant before the
        cutover still returns A; after, it returns B.
        """

    # --- reads ---
    @abstractmethod
    def get_memory(self, memory_id: str) -> Memory | None: ...

    @abstractmethod
    def vector_candidates(
        self, embedding: list[float], k: int, types: list[MemoryType] | None = None
    ) -> list[Candidate]:
        """Top-k by vector similarity (ranked)."""

    @abstractmethod
    def lexical_candidates(
        self, query: str, k: int, types: list[MemoryType] | None = None
    ) -> list[Candidate]:
        """Top-k by FTS5/BM25 lexical match (ranked)."""

    @abstractmethod
    def edges_for(self, memory_id: str, as_of: datetime | None = None) -> list[Edge]:
        """Edges touching a memory, optionally filtered to those active at `as_of`."""

    @abstractmethod
    def timeline(self, entity_id: str) -> list[Edge]:
        """All edges touching an entity, ordered by valid_from (full history)."""
