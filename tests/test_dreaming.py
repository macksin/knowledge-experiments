"""The dreaming phase: raw episodes consolidate into durable, separated memory types."""

from __future__ import annotations

import pytest

from knowledge_mcp.dreaming.consolidator import HeuristicConsolidator
from knowledge_mcp.dreaming.engine import DreamEngine
from knowledge_mcp.models import ConsolidationState, MemoryType


@pytest.fixture
def dream(engine):
    return DreamEngine(engine, consolidator=HeuristicConsolidator())


def test_heuristic_classifies_the_three_types():
    c = HeuristicConsolidator()
    assert c.classify("I always prefer dark mode in my editor.") == MemoryType.SELF_SCHEMA
    assert c.classify("When the build fails, then rerun the release script.") == (
        MemoryType.PROCEDURAL
    )
    assert c.classify("The capital of France is Paris.") == MemoryType.SEMANTIC


def test_dream_separates_the_stream_and_links_provenance(engine, dream):
    pref = engine.remember("I always prefer dark mode in my editor.", type=MemoryType.EPISODIC)
    proc = engine.remember(
        "When the build fails, then rerun the release script.", type=MemoryType.EPISODIC
    )
    fact = engine.remember("The capital of France is Paris.", type=MemoryType.EPISODIC)

    report = dream.dream()

    assert report.episodes_processed == 3
    assert report.derived_by_type == {"self_schema": 1, "procedural": 1, "semantic": 1}

    # Each derived memory links back to its source episode (DERIVED_FROM provenance).
    for source in (pref, proc, fact):
        edges = engine.store.edges_for(source)
        assert any(e.relation == "DERIVED_FROM" and e.to_id == source for e in edges)

    # The derived self_schema memory is recallable as its consolidated type.
    hits = engine.recall("dark mode editor preference", types=[MemoryType.SELF_SCHEMA], k=5)
    assert hits


def test_dream_marks_episodes_consolidated_and_is_idempotent(engine, dream):
    engine.remember("The capital of France is Paris.", type=MemoryType.EPISODIC)
    engine.remember("I love using vim keybindings.", type=MemoryType.EPISODIC)

    first = dream.dream()
    assert first.episodes_processed == 2

    # Episodes are now consolidated; a second pass finds nothing new.
    pending = engine.store.list_memories(
        types=[MemoryType.EPISODIC], state=ConsolidationState.RAW
    )
    assert pending == []
    second = dream.dream()
    assert second.episodes_processed == 0
    assert second.derived_ids == []


def test_dream_status_reports_pending_and_counts(engine, dream):
    engine.remember("The capital of France is Paris.", type=MemoryType.EPISODIC)
    before = dream.status()
    assert before["pending_episodes"] == 1

    dream.dream()
    after = dream.status()
    assert after["pending_episodes"] == 0
    assert after["by_type"].get("semantic", 0) >= 1
