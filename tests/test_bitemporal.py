"""The canonical 'project focus A -> B' bitemporal case.

A project focuses on A from t0; at the cutover t1 focus shifts to B. This is a
valid-time change, so we never delete the A edge -- we close its valid-time only
(it stays believed) and add the B edge. A point-in-time query must return A before
t1 and B after t1, with both rows retained forever.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from knowledge_mcp.models import Edge, Memory, MemoryType


def _mem(store, mid: str, content: str) -> str:
    store.add_memory(
        Memory(id=mid, type=MemoryType.SEMANTIC, content=content), [0.0] * 384
    )
    return mid


def test_project_focus_a_then_b(store):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 3, 1, tzinfo=UTC)

    _mem(store, "project", "Project Helix")
    _mem(store, "A", "objective A")
    _mem(store, "B", "objective B")

    # Focus on A, valid from t0.
    edge_a = Edge(
        id="e_a", from_id="project", to_id="A", relation="FOCUSES_ON", valid_from=t0
    )
    store.add_edge(edge_a)

    # Cutover at t1: the focus genuinely shifted (valid-time change), so close A's
    # valid-time only -- it stays believed -- and open B.
    store.close_edge("e_a", valid_to=t1)
    store.add_edge(
        Edge(id="e_b", from_id="project", to_id="B", relation="FOCUSES_ON", valid_from=t1)
    )

    before = store.edges_for("project", as_of=t0 + timedelta(days=10))
    after = store.edges_for("project", as_of=t1 + timedelta(days=10))

    # Before the cutover, the active edge points to A...
    assert {e.to_id for e in before} == {"A"}
    # ...after the cutover it points to B.
    assert {e.to_id for e in after} == {"B"}

    # Nothing was deleted: full history retains both edges, ordered by valid_from.
    history = store.timeline("project")
    assert [e.to_id for e in history] == ["A", "B"]


def test_unfiltered_edges_return_all(store):
    _mem(store, "n1", "n1")
    _mem(store, "n2", "n2")
    store.add_edge(Edge(id="x", from_id="n1", to_id="n2", relation="MENTIONS"))
    assert len(store.edges_for("n1")) == 1
