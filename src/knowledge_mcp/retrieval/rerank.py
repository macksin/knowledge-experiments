"""Reranker provider: identity fallback + optional cross-encoder.

`IdentityReranker` preserves the fused RRF order (offline default).
`CrossEncoderReranker` rescores (query, content) pairs with bge-reranker-v2-m3
when sentence-transformers + the model are available.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from knowledge_mcp.config import settings

# A rerank input/output item: (memory_id, text, score).
RankItem = tuple[str, str, float]


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, items: list[RankItem], top_k: int) -> list[RankItem]: ...


class IdentityReranker:
    """Keep incoming order (already RRF-ranked); just truncate to top_k."""

    def rerank(self, query: str, items: list[RankItem], top_k: int) -> list[RankItem]:
        return items[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder  # lazy optional import

        self._model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(self, query: str, items: list[RankItem], top_k: int) -> list[RankItem]:
        if not items:
            return []
        scores = self._model.predict([(query, text) for _, text, _ in items])
        rescored = [
            (mid, text, float(s)) for (mid, text, _), s in zip(items, scores, strict=True)
        ]
        rescored.sort(key=lambda x: x[2], reverse=True)
        return rescored[:top_k]


def default_reranker() -> Reranker:
    if settings.cross_encoder_available:
        try:
            return CrossEncoderReranker()
        except Exception:
            pass
    return IdentityReranker()
