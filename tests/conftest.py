"""Shared fixtures. Everything uses the offline providers (HashEmbedder /
IdentityReranker / NoOpContextualizer) so tests are deterministic and network-free.
"""

from __future__ import annotations

import pytest

from knowledge_mcp.engine import KnowledgeEngine
from knowledge_mcp.retrieval.contextual import NoOpContextualizer
from knowledge_mcp.retrieval.embed import HashEmbedder
from knowledge_mcp.retrieval.rerank import IdentityReranker
from knowledge_mcp.store.sqlite_store import SQLiteStore

DIM = 384


@pytest.fixture
def store(tmp_path) -> SQLiteStore:
    s = SQLiteStore(str(tmp_path / "test.db"), dim=DIM)
    s.init_schema()
    yield s
    s.close()


@pytest.fixture
def engine(store) -> KnowledgeEngine:
    return KnowledgeEngine(
        store=store,
        embedder=HashEmbedder(dim=DIM),
        reranker=IdentityReranker(),
        contextualizer=NoOpContextualizer(),
    )
