"""MemoryStore conformance suite.

This is the contract every backend must satisfy. In Phase 4 the Postgres store is
added to `store_factories` and must pass this file unchanged -- that is what makes
the SQLite->Postgres graduation a swap, not a rewrite.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from knowledge_mcp.models import Edge, Memory, MemoryType
from knowledge_mcp.store.sqlite_store import SQLiteStore

DIM = 384


def _sqlite_factory(tmp_path):
    s = SQLiteStore(str(tmp_path / "conf.db"), dim=DIM)
    s.init_schema()
    return s


# Each backend registers a factory here. Phase 4: append a Postgres factory.
store_factories = [_sqlite_factory]


@pytest.fixture(params=store_factories)
def any_store(request, tmp_path):
    s = request.param(tmp_path)
    yield s
    if hasattr(s, "close"):
        s.close()


def _vec(seed: float) -> list[float]:
    return [seed] + [0.0] * (DIM - 1)


def test_add_and_get_memory_roundtrip(any_store):
    mem = Memory(id="m1", type=MemoryType.PROCEDURAL, content="how to deploy", source="docs")
    any_store.add_memory(mem, _vec(1.0))
    got = any_store.get_memory("m1")
    assert got is not None
    assert got.content == "how to deploy"
    assert got.type == MemoryType.PROCEDURAL
    assert got.source == "docs"


def test_get_missing_returns_none(any_store):
    assert any_store.get_memory("nope") is None


def test_lexical_candidates_match(any_store):
    any_store.add_memory(Memory(id="m1", content="alpha bravo charlie"), _vec(1.0))
    any_store.add_memory(Memory(id="m2", content="delta echo foxtrot"), _vec(2.0))
    cands = any_store.lexical_candidates("bravo", k=10)
    ids = [mid for mid, _ in cands]
    assert "m1" in ids and "m2" not in ids


def test_vector_candidates_ranked(any_store):
    any_store.add_memory(Memory(id="near", content="x"), _vec(1.0))
    any_store.add_memory(Memory(id="far", content="y"), _vec(-1.0))
    cands = any_store.vector_candidates(_vec(1.0), k=10)
    assert cands[0][0] == "near"


def test_type_filter_on_candidates(any_store):
    any_store.add_memory(Memory(id="s", type=MemoryType.SEMANTIC, content="paris"), _vec(1.0))
    any_store.add_memory(Memory(id="e", type=MemoryType.EPISODIC, content="paris"), _vec(1.0))
    ids = [m for m, _ in any_store.lexical_candidates("paris", 10, [MemoryType.SEMANTIC])]
    assert ids == ["s"]


def test_bitemporal_edge_lifecycle(any_store):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 6, 1, tzinfo=UTC)
    any_store.add_memory(Memory(id="p", content="p"), _vec(1.0))
    any_store.add_memory(Memory(id="a", content="a"), _vec(1.0))
    any_store.add_edge(Edge(id="e1", from_id="p", to_id="a", relation="FOCUSES_ON", valid_from=t0))

    active = any_store.edges_for("p", as_of=t0 + timedelta(days=1))
    assert len(active) == 1

    # Valid-time close: not active after t1, but still believed and in history.
    any_store.close_edge("e1", valid_to=t1)
    assert any_store.edges_for("p", as_of=t1 + timedelta(days=1)) == []
    assert any_store.edges_for("p", as_of=t0 + timedelta(days=1))  # still active before t1
    assert len(any_store.timeline("p")) == 1


def test_retraction_drops_from_current_belief(any_store):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    any_store.add_memory(Memory(id="p", content="p"), _vec(1.0))
    any_store.add_memory(Memory(id="a", content="a"), _vec(1.0))
    any_store.add_edge(Edge(id="e1", from_id="p", to_id="a", relation="FOCUSES_ON", valid_from=t0))

    # Transaction-time retraction: gone from point-in-time reads, retained in history.
    any_store.close_edge("e1", expired_at=t0 + timedelta(days=2))
    assert any_store.edges_for("p", as_of=t0 + timedelta(days=1)) == []
    assert len(any_store.timeline("p")) == 1


def test_embedding_dim_mismatch_raises(any_store):
    with pytest.raises(ValueError):
        any_store.add_memory(Memory(id="bad", content="x"), [0.0, 0.0])
