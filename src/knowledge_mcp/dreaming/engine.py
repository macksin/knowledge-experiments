"""The dreaming phase: the nightly slow path that consolidates memory.

This is the conceptual payoff of the project. Capture (`KnowledgeEngine.remember`)
is deliberately dumb: everything lands as a raw, time-stamped episode. The dreaming
job is where reasoning happens and where the single stream *separates* into the
durable memory types -- which is the answer to "are facts, things-I-learned, and my
preferences the same thing?": they enter together and consolidation pulls them apart.

`dream()` is idempotent: it only scans raw episodes and marks them consolidated as it
goes, so re-running processes nothing new. Each derived memory is captured through the
normal `remember` path (so it is contextualized, embedded and indexed like any other)
and linked back to its source episode with a `DERIVED_FROM` edge for full provenance.

Scheduling is intentionally out of scope here: `dream()` is the callable a cron entry
or APScheduler job invokes nightly, and `dream_now()` exposes it for manual triggering.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowledge_mcp.dreaming.consolidator import Consolidator, default_consolidator
from knowledge_mcp.engine import KnowledgeEngine
from knowledge_mcp.models import ConsolidationState, MemoryType, Relation


class DreamReport(BaseModel):
    """Summary of one dreaming run."""

    episodes_processed: int = 0
    derived_by_type: dict[str, int] = Field(default_factory=dict)
    derived_ids: list[str] = Field(default_factory=list)


class DreamEngine:
    def __init__(
        self,
        engine: KnowledgeEngine,
        consolidator: Consolidator | None = None,
    ) -> None:
        self.engine = engine
        self.store = engine.store
        self.consolidator = consolidator or default_consolidator()

    def dream(self, batch_limit: int | None = None) -> DreamReport:
        """Consolidate raw episodes into durable memories. Idempotent."""
        episodes = self.store.list_memories(
            types=[MemoryType.EPISODIC],
            state=ConsolidationState.RAW,
            limit=batch_limit,
        )
        report = DreamReport()
        for episode in episodes:
            for proposal in self.consolidator.consolidate(episode):
                # Defensive: never emit another raw episode (would loop on re-run).
                if proposal.type == MemoryType.EPISODIC:
                    continue
                derived_id = self.engine.remember(
                    content=proposal.content,
                    type=proposal.type,
                    source=episode.source,
                    links=[{"to": episode.id, "relation": Relation.DERIVED_FROM.value}],
                )
                self.store.set_consolidation_state(derived_id, ConsolidationState.CONSOLIDATED)
                key = proposal.type.value
                report.derived_by_type[key] = report.derived_by_type.get(key, 0) + 1
                report.derived_ids.append(derived_id)
            # Mark the source episode processed so the next run skips it.
            self.store.set_consolidation_state(episode.id, ConsolidationState.CONSOLIDATED)
            report.episodes_processed += 1
        return report

    def status(self) -> dict:
        """Lightweight introspection: what is pending and what has been consolidated."""
        pending = self.store.list_memories(
            types=[MemoryType.EPISODIC], state=ConsolidationState.RAW
        )
        all_memories = self.store.list_memories()
        by_type: dict[str, int] = {}
        for m in all_memories:
            by_type[m.type.value] = by_type.get(m.type.value, 0) + 1
        return {
            "pending_episodes": len(pending),
            "total_memories": len(all_memories),
            "by_type": by_type,
        }
