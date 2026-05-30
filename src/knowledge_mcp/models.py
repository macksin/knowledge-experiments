"""Core domain models: the four-type memory taxonomy, memories, edges, results.

The taxonomy is the conceptual spine of the project. The user's three intuitive
buckets ("facts", "things I learned along the way", "how I think") map onto four
cognitive memory systems plus a consolidation *process*:

    semantic    -- knowing *that*  : decontextualized facts about the world
    episodic    -- knowing *when*  : time-stamped personal events with provenance
    procedural  -- knowing *how*   : skills / heuristics ("when X, do Z")
    self_schema -- knowing *who*   : preferences, values, mental models (the user)

"Things learned along the way" is not a type -- it is the dreaming process that
consolidates raw episodes into the other three types. Everything therefore enters
as `episodic`; consolidation (Phase 3) separates the stream.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware UTC now -- used as the default transaction-time stamp."""
    return datetime.now(UTC)


class MemoryType(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    SELF_SCHEMA = "self_schema"


class ConsolidationState(StrEnum):
    RAW = "raw"
    CONSOLIDATED = "consolidated"


# Typed graph relations. Open set -- these are the relations the system reasons
# about today; `link()` accepts arbitrary strings, but these are the documented ones.
class Relation(StrEnum):
    FOCUSES_ON = "FOCUSES_ON"
    LEADS = "LEADS"
    CAUSES = "CAUSES"
    BEFORE = "BEFORE"
    MENTIONS = "MENTIONS"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    ANNOTATES = "ANNOTATES"
    # Provenance: a consolidated memory was derived from a raw source episode (dreaming).
    DERIVED_FROM = "DERIVED_FROM"


class Memory(BaseModel):
    """A single memory/node.

    Bitemporal fields come in two pairs:
      * valid-time      (`valid_from` / `valid_to`): when the fact is true *in the world*.
      * transaction-time(`created_at` / `expired_at`): when the row was believed *by the system*.
    Invalidation never deletes: an outdated row gets `expired_at` set and a new row is added.
    """

    id: str
    type: MemoryType = MemoryType.EPISODIC
    content: str
    source: str | None = None
    confidence: float = 1.0
    consolidation_state: ConsolidationState = ConsolidationState.RAW

    # valid-time
    valid_from: datetime = Field(default_factory=utcnow)
    valid_to: datetime | None = None
    # transaction-time
    created_at: datetime = Field(default_factory=utcnow)
    expired_at: datetime | None = None

    # Cached contextual blurb prepended before indexing (Anthropic Contextual Retrieval).
    contextual_blurb: str = ""


class Edge(BaseModel):
    """A typed, bitemporal relation between two memories/entities."""

    id: str
    from_id: str
    to_id: str
    relation: str

    valid_from: datetime = Field(default_factory=utcnow)
    valid_to: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    expired_at: datetime | None = None


class RecallHit(BaseModel):
    """A recall result: the memory, its fused/reranked score, and graph context."""

    memory: Memory
    score: float
    graph_context: list[Edge] = Field(default_factory=list)
