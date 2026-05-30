"""The fast-path engine: capture & retrieve. Transport-agnostic so it can be
driven directly by tests and wrapped by the FastMCP server.

`remember` captures everything as a time-stamped memory (defaulting to episodic),
contextualizes it against its links, embeds, and indexes. `recall` runs the hybrid
contextual pipeline: vector + BM25 candidates -> RRF -> rerank -> attach graph
context. `link` writes bitemporal edges. `timeline` exposes point-in-time history.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from knowledge_mcp.models import Edge, Memory, MemoryType, RecallHit, Relation, utcnow
from knowledge_mcp.retrieval.contextual import Contextualizer, default_contextualizer
from knowledge_mcp.retrieval.embed import Embedder, default_embedder
from knowledge_mcp.retrieval.hybrid import rrf_fuse
from knowledge_mcp.retrieval.rerank import RankItem, Reranker, default_reranker
from knowledge_mcp.store.base import MemoryStore

# Over-retrieve wide, fuse, rerank to a small final set (Anthropic Contextual Retrieval).
CANDIDATES_PER_RETRIEVER = 150
RERANK_POOL = 20


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class KnowledgeEngine:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        contextualizer: Contextualizer | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder or default_embedder()
        self.reranker = reranker or default_reranker()
        self.contextualizer = contextualizer or default_contextualizer()

    # ----------------------------------------------------------------- capture
    def remember(
        self,
        content: str,
        type: MemoryType | str = MemoryType.EPISODIC,
        source: str | None = None,
        occurred_at: datetime | None = None,
        links: list[dict] | None = None,
    ) -> str:
        mem_type = MemoryType(type) if isinstance(type, str) else type
        now = utcnow()
        memory = Memory(
            id=_new_id("mem"),
            type=mem_type,
            content=content,
            source=source,
            valid_from=occurred_at or now,
            created_at=now,
        )

        # Contextualize against any neighbors named in `links` (graph-as-context).
        neighbors: list[tuple[Edge, Memory]] = []
        for spec in links or []:
            neighbor = self.store.get_memory(spec["to"])
            if neighbor is not None:
                stub_edge = Edge(
                    id="pending",
                    from_id=memory.id,
                    to_id=neighbor.id,
                    relation=spec.get("relation", "MENTIONS"),
                )
                neighbors.append((stub_edge, neighbor))
        memory.contextual_blurb = self.contextualizer.contextualize(memory, neighbors)

        index_text = (
            f"{memory.contextual_blurb}\n{content}" if memory.contextual_blurb else content
        )
        embedding = self.embedder.embed([index_text])[0]
        self.store.add_memory(memory, embedding)

        # Persist the requested links as real bitemporal edges.
        for spec in links or []:
            self.link(
                memory.id,
                spec["to"],
                spec.get("relation", "MENTIONS"),
                valid_from=memory.valid_from,
            )
        return memory.id

    # --------------------------------------------------------------- retrieval
    def recall(
        self,
        query: str,
        as_of: datetime | None = None,
        types: list[MemoryType] | None = None,
        k: int = 5,
    ) -> list[RecallHit]:
        q_emb = self.embedder.embed([query])[0]
        vector = self.store.vector_candidates(q_emb, CANDIDATES_PER_RETRIEVER, types)
        lexical = self.store.lexical_candidates(query, CANDIDATES_PER_RETRIEVER, types)

        fused = rrf_fuse([vector, lexical], top_n=RERANK_POOL)

        items: list[RankItem] = []
        memories: dict[str, Memory] = {}
        for mid, score in fused:
            mem = self.store.get_memory(mid)
            if mem is None:
                continue
            memories[mid] = mem
            items.append((mid, mem.content, score))

        reranked = self.reranker.rerank(query, items, top_k=k)

        hits: list[RecallHit] = []
        for mid, _text, score in reranked:
            mem = memories[mid]
            hits.append(
                RecallHit(
                    memory=mem,
                    score=score,
                    graph_context=self.store.edges_for(mid, as_of=as_of),
                )
            )
        return hits

    # --------------------------------------------------------------- graph ops
    def link(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        supersede: bool = False,
    ) -> str:
        """Write a bitemporal edge.

        With `supersede=True` this is a contradiction-aware update: any currently-open
        edge with the same `(from_id, relation)` but a *different* target is bitemporally
        closed in valid-time at the new edge's `valid_from` before the new edge is added.
        This is what turns the canonical "project focus A -> B" shift into a single call
        (the old A edge stays queryable for instants before the cutover; B applies after).
        Both rows live forever -- nothing is deleted.
        """
        vf = valid_from or utcnow()
        if supersede:
            for e in self.store.edges_for(from_id):
                if (
                    e.from_id == from_id
                    and e.relation == relation
                    and e.to_id != to_id
                    and e.valid_to is None
                    and e.expired_at is None
                ):
                    self.store.close_edge(e.id, valid_to=vf)
        edge = Edge(
            id=_new_id("edge"),
            from_id=from_id,
            to_id=to_id,
            relation=relation,
            valid_from=vf,
            valid_to=valid_to,
        )
        return self.store.add_edge(edge)

    def annotate(self, target_id: str, note: str, source: str | None = None) -> str:
        """Attach a first-class annotation to an existing memory.

        The annotation is itself a memory (captured, contextualized against its target,
        embedded and indexed like any other) linked back to the target with an
        `ANNOTATES` edge, so it is both recallable and visible in the target's timeline.
        """
        if self.store.get_memory(target_id) is None:
            raise ValueError(f"unknown memory: {target_id}")
        return self.remember(
            content=note,
            type=MemoryType.EPISODIC,
            source=source,
            links=[{"to": target_id, "relation": Relation.ANNOTATES.value}],
        )

    def timeline(self, entity_id: str) -> list[Edge]:
        return self.store.timeline(entity_id)
