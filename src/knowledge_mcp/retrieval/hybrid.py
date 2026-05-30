"""Reciprocal Rank Fusion (RRF) over multiple ranked candidate lists.

RRF combines rankings without needing comparable scores across retrievers: each
list contributes 1/(k + rank) to an item's fused score. k≈60 (Cormack et al.) damps
the contribution of low ranks. We fuse the vector and BM25 candidate lists, then
hand the top fused ids to the reranker.
"""

from __future__ import annotations

from knowledge_mcp.store.base import Candidate

RRF_K = 60


def rrf_fuse(
    ranked_lists: list[list[Candidate]], k: int = RRF_K, top_n: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked (id, score) lists into one ranking by reciprocal rank.

    Only the *order* within each list matters (the per-list scores are ignored),
    which is exactly what lets us mix a distance-based and a BM25-based ranking.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (mid, _score) in enumerate(ranked):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:top_n] if top_n is not None else ordered
