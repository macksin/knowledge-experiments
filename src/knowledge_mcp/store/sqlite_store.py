"""SQLite implementation of `MemoryStore` (the MVP backend).

Uses three coordinated tables/vtabs sharing the memory id as rowid linkage:
  * `memories`      -- canonical rows (bitemporal + taxonomy fields)
  * `memories_fts`  -- FTS5 virtual table over content + contextual_blurb (BM25)
  * `memories_vec`  -- sqlite-vec vec0 virtual table for kNN

FTS5 and vec0 use an INTEGER `rowid`; memory ids are opaque strings, so we keep a
stable string<->int mapping via the `memories.rowid` column.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime

import sqlite_vec

from knowledge_mcp.config import settings
from knowledge_mcp.graph.queries import ACTIVE_AS_OF
from knowledge_mcp.models import (
    ConsolidationState,
    Edge,
    Memory,
    MemoryType,
)
from knowledge_mcp.store.base import Candidate, MemoryStore

_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _fts_match_query(query: str) -> str | None:
    """Turn a free-text query into a safe FTS5 MATCH expression.

    Raw user queries can contain FTS5 operators (?, ", *, :, parentheses, AND/OR/NOT),
    which raise `fts5: syntax error`. We extract plain word tokens, quote each as a
    string literal, and OR them so lexical search broadens the candidate pool (the
    reranker handles precision). Returns None when the query has no usable tokens.
    """
    tokens = _FTS_TOKEN_RE.findall(query)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class SQLiteStore(MemoryStore):
    def __init__(self, db_path: str | None = None, dim: int | None = None) -> None:
        self.db_path = db_path or settings.db_path
        self.dim = dim or settings.embedding_dim
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------ schema
    def init_schema(self) -> None:
        c = self.conn
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                consolidation_state TEXT NOT NULL DEFAULT 'raw',
                contextual_blurb TEXT NOT NULL DEFAULT '',
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                created_at TEXT NOT NULL,
                expired_at TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                created_at TEXT NOT NULL,
                expired_at TEXT
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id)")
        c.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, contextual_blurb, content=''
            )
            """
        )
        c.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                embedding float[{self.dim}]
            )
            """
        )
        c.commit()

    # ------------------------------------------------------------------ writes
    def add_memory(self, memory: Memory, embedding: list[float]) -> str:
        if len(embedding) != self.dim:
            raise ValueError(f"embedding dim {len(embedding)} != configured {self.dim}")
        c = self.conn
        cur = c.execute(
            """
            INSERT INTO memories
                (id, type, content, source, confidence, consolidation_state,
                 contextual_blurb, valid_from, valid_to, created_at, expired_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                memory.id,
                memory.type.value,
                memory.content,
                memory.source,
                memory.confidence,
                memory.consolidation_state.value,
                memory.contextual_blurb,
                _iso(memory.valid_from),
                _iso(memory.valid_to),
                _iso(memory.created_at),
                _iso(memory.expired_at),
            ),
        )
        rowid = cur.lastrowid
        c.execute(
            "INSERT INTO memories_fts(rowid, content, contextual_blurb) VALUES (?,?,?)",
            (rowid, memory.content, memory.contextual_blurb),
        )
        c.execute(
            "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, sqlite_vec.serialize_float32(embedding)),
        )
        c.commit()
        return memory.id

    def add_edge(self, edge: Edge) -> str:
        self.conn.execute(
            """
            INSERT INTO edges (id, from_id, to_id, relation,
                               valid_from, valid_to, created_at, expired_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                edge.id,
                edge.from_id,
                edge.to_id,
                edge.relation,
                _iso(edge.valid_from),
                _iso(edge.valid_to),
                _iso(edge.created_at),
                _iso(edge.expired_at),
            ),
        )
        self.conn.commit()
        return edge.id

    def close_edge(
        self,
        edge_id: str,
        valid_to: datetime | None = None,
        expired_at: datetime | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[object] = []
        if valid_to is not None:
            sets.append("valid_to = ?")
            params.append(_iso(valid_to))
        if expired_at is not None:
            sets.append("expired_at = ?")
            params.append(_iso(expired_at))
        if not sets:
            return
        params.append(edge_id)
        self.conn.execute(f"UPDATE edges SET {', '.join(sets)} WHERE id = ?", params)
        self.conn.commit()

    # ------------------------------------------------------------------- reads
    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            consolidation_state=ConsolidationState(row["consolidation_state"]),
            contextual_blurb=row["contextual_blurb"],
            valid_from=_dt(row["valid_from"]),
            valid_to=_dt(row["valid_to"]),
            created_at=_dt(row["created_at"]),
            expired_at=_dt(row["expired_at"]),
        )

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def _type_filter(self, types: list[MemoryType] | None) -> tuple[str, list[str]]:
        if not types:
            return "", []
        placeholders = ",".join("?" for _ in types)
        return f" AND m.type IN ({placeholders})", [t.value for t in types]

    def vector_candidates(
        self, embedding: list[float], k: int, types: list[MemoryType] | None = None
    ) -> list[Candidate]:
        type_clause, type_params = self._type_filter(types)
        # vec0 KNN returns distance (smaller == closer); convert to a descending score.
        rows = self.conn.execute(
            f"""
            SELECT m.id AS id, v.distance AS distance
            FROM memories_vec v
            JOIN memories m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?{type_clause}
            ORDER BY v.distance
            """,
            [sqlite_vec.serialize_float32(embedding), k, *type_params],
        ).fetchall()
        return [(r["id"], -float(r["distance"])) for r in rows]

    def lexical_candidates(
        self, query: str, k: int, types: list[MemoryType] | None = None
    ) -> list[Candidate]:
        match = _fts_match_query(query)
        if match is None:
            return []
        type_clause, type_params = self._type_filter(types)
        # FTS5 bm25() is smaller-is-better; negate for a descending score.
        rows = self.conn.execute(
            f"""
            SELECT m.id AS id, bm25(memories_fts) AS rank
            FROM memories_fts
            JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ?{type_clause}
            ORDER BY rank
            LIMIT ?
            """,
            [match, *type_params, k],
        ).fetchall()
        return [(r["id"], -float(r["rank"])) for r in rows]

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        return Edge(
            id=row["id"],
            from_id=row["from_id"],
            to_id=row["to_id"],
            relation=row["relation"],
            valid_from=_dt(row["valid_from"]),
            valid_to=_dt(row["valid_to"]),
            created_at=_dt(row["created_at"]),
            expired_at=_dt(row["expired_at"]),
        )

    def edges_for(self, memory_id: str, as_of: datetime | None = None) -> list[Edge]:
        sql = "SELECT * FROM edges WHERE (from_id = :id OR to_id = :id)"
        params: dict[str, object] = {"id": memory_id}
        if as_of is not None:
            sql += f" AND {ACTIVE_AS_OF}"
            params["as_of"] = _iso(as_of)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def timeline(self, entity_id: str) -> list[Edge]:
        rows = self.conn.execute(
            """
            SELECT * FROM edges
            WHERE from_id = :id OR to_id = :id
            ORDER BY valid_from ASC
            """,
            {"id": entity_id},
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def list_memories(
        self,
        types: list[MemoryType] | None = None,
        state: ConsolidationState | None = None,
        limit: int | None = None,
    ) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[object] = []
        if types:
            placeholders = ",".join("?" for _ in types)
            sql += f" AND type IN ({placeholders})"
            params.extend(t.value for t in types)
        if state is not None:
            sql += " AND consolidation_state = ?"
            params.append(state.value)
        sql += " ORDER BY created_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def set_consolidation_state(self, memory_id: str, state: ConsolidationState) -> None:
        self.conn.execute(
            "UPDATE memories SET consolidation_state = ? WHERE id = ?",
            (state.value, memory_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
