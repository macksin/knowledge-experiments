"""Embedder provider: deterministic offline fallback + optional real model.

`HashEmbedder` is a network-free, deterministic embedder: it hashes tokens into a
fixed-dimension bag-of-features vector and L2-normalizes. It is *not* semantically
strong, but it is stable and lets the entire pipeline (indexing, vector kNN, RRF,
tests) run with no model download. `FastEmbedEmbedder` swaps in real bge embeddings
when `fastembed` is installed and HuggingFace is reachable.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from knowledge_mcp.config import settings

_TOKEN_RE = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic hashing embedder (offline default)."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class FastEmbedEmbedder:
    """Real embeddings via fastembed (bge-small, 384-dim). Lazily initialized."""

    def __init__(self, model_name: str | None = None, dim: int | None = None) -> None:
        from fastembed import TextEmbedding  # imported lazily; optional dep

        self.dim = dim or settings.embedding_dim
        self._model = TextEmbedding(model_name=model_name or settings.embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


def default_embedder() -> Embedder:
    """Pick the best available embedder. Falls back to HashEmbedder offline."""
    if settings.fastembed_available:
        try:
            return FastEmbedEmbedder()
        except Exception:
            pass
    return HashEmbedder()
