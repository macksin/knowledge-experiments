"""FastMCP server exposing the Phase 1 fast-path tools over stdio.

The tools are thin wrappers over `KnowledgeEngine`; all logic lives there so it can
be tested without the MCP transport. Providers default to the offline fallbacks, so
the server runs with no API key and no model downloads.
"""

from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP

from knowledge_mcp.config import settings
from knowledge_mcp.engine import KnowledgeEngine
from knowledge_mcp.models import MemoryType
from knowledge_mcp.store.sqlite_store import SQLiteStore

mcp = FastMCP("knowledge-mcp")

_engine: KnowledgeEngine | None = None


def get_engine() -> KnowledgeEngine:
    global _engine
    if _engine is None:
        store = SQLiteStore(settings.db_path)
        store.init_schema()
        _engine = KnowledgeEngine(store)
    return _engine


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@mcp.tool()
def remember(
    content: str,
    type: str = "episodic",
    source: str | None = None,
    occurred_at: str | None = None,
    links: list[dict] | None = None,
) -> dict:
    """Store a memory (defaults to episodic) with provenance + timestamp.

    `links` is a list of {"to": memory_id, "relation": REL} dicts.
    Returns the new memory id.
    """
    mem_id = get_engine().remember(
        content=content,
        type=MemoryType(type),
        source=source,
        occurred_at=_parse_dt(occurred_at),
        links=links,
    )
    return {"id": mem_id}


@mcp.tool()
def recall(
    query: str,
    as_of: str | None = None,
    types: list[str] | None = None,
    k: int = 5,
) -> list[dict]:
    """Hybrid contextual retrieval. Optional point-in-time `as_of` (ISO-8601)
    filters the attached graph context. Returns memories + scores + graph context.
    """
    hits = get_engine().recall(
        query=query,
        as_of=_parse_dt(as_of),
        types=[MemoryType(t) for t in types] if types else None,
        k=k,
    )
    return [h.model_dump(mode="json") for h in hits]


@mcp.tool()
def link(
    from_id: str,
    to_id: str,
    relation: str,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> dict:
    """Create a typed bitemporal edge between two memories/entities."""
    edge_id = get_engine().link(
        from_id=from_id,
        to_id=to_id,
        relation=relation,
        valid_from=_parse_dt(valid_from),
        valid_to=_parse_dt(valid_to),
    )
    return {"id": edge_id}


@mcp.tool()
def timeline(entity_or_topic: str) -> list[dict]:
    """History / point-in-time view of an entity's edges (the A->B case)."""
    edges = get_engine().timeline(entity_or_topic)
    return [e.model_dump(mode="json") for e in edges]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
