"""End-to-end fast-path exercise through the engine (what the MCP tools call)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from knowledge_mcp.models import MemoryType


def test_remember_then_recall(engine):
    engine.remember("The capital of France is Paris.", type=MemoryType.SEMANTIC)
    engine.remember("I prefer dark mode in my editor.", type=MemoryType.SELF_SCHEMA)
    engine.remember("Deploy by running the release script.", type=MemoryType.PROCEDURAL)

    hits = engine.recall("capital France Paris", k=3)
    assert hits
    assert "Paris" in hits[0].memory.content


def test_recall_type_filter(engine):
    engine.remember("Paris is in France.", type=MemoryType.SEMANTIC)
    engine.remember("I like Paris in the spring.", type=MemoryType.SELF_SCHEMA)
    hits = engine.recall("Paris", types=[MemoryType.SELF_SCHEMA], k=5)
    assert hits
    assert all(h.memory.type == MemoryType.SELF_SCHEMA for h in hits)


def test_remember_with_links_creates_edges(engine):
    proj = engine.remember("Project Helix kickoff.", type=MemoryType.EPISODIC)
    engine.remember(
        "Decided objective A.",
        type=MemoryType.EPISODIC,
        links=[{"to": proj, "relation": "MENTIONS"}],
    )
    assert len(engine.timeline(proj)) == 1


def test_recall_handles_fts5_operator_chars_in_query(engine):
    """Regression: raw queries with FTS5 operators (?, ", *, :, parens) must not
    raise `fts5: syntax error`. The query is sanitized into quoted tokens.
    """
    engine.remember("Helios focuses on retrieval quality.", type=MemoryType.SEMANTIC)

    for query in [
        'what does Helios focus on?',
        'retrieval "quality"',
        'Helios: (focus) * AND',
        '???',  # no usable tokens -> lexical contributes nothing, must not crash
    ]:
        hits = engine.recall(query, k=3)  # must not raise
        assert isinstance(hits, list)

    # The substantive query still finds the memory.
    hits = engine.recall("what does Helios focus on?", k=3)
    assert any("Helios" in h.memory.content for h in hits)


def test_recall_attaches_pointintime_graph_context(engine):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 3, 1, tzinfo=UTC)
    proj = engine.remember("Project Helix.", type=MemoryType.SEMANTIC, occurred_at=t0)
    a = engine.remember("Objective A.", type=MemoryType.SEMANTIC, occurred_at=t0)
    edge_id = engine.link(proj, a, "FOCUSES_ON", valid_from=t0)
    engine.store.close_edge(edge_id, valid_to=t1)

    hits = engine.recall("Helix", as_of=t0 + timedelta(days=5), k=5)
    proj_hit = next(h for h in hits if h.memory.id == proj)
    assert any(e.to_id == a for e in proj_hit.graph_context)

    hits_after = engine.recall("Helix", as_of=t1 + timedelta(days=5), k=5)
    proj_hit_after = next(h for h in hits_after if h.memory.id == proj)
    assert all(e.to_id != a for e in proj_hit_after.graph_context)
