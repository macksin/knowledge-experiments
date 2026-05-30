"""Settings + provider auto-selection.

The MVP is offline-first: it runs with deterministic fallback providers and no
network. When real backends are importable (and, for Anthropic, an API key is
present), the factory functions transparently upgrade. Tests inject providers
explicitly and so never depend on this auto-detection.
"""

from __future__ import annotations

import importlib.util
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KNOWLEDGE_", extra="ignore")

    db_path: str = "./knowledge.db"

    # Embedding config. embedding_dim must match the active embedder's output and
    # the vec0 virtual-table declaration; HashEmbedder and FastEmbed's bge-small
    # both emit 384-dim vectors so the schema is stable across the offline/real swap.
    embedding_dim: int = 384
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Anthropic (contextual blurbs)
    anthropic_model: str = "claude-haiku-4-5-20251001"

    @property
    def anthropic_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY")) and _module_available("anthropic")

    @property
    def fastembed_available(self) -> bool:
        return _module_available("fastembed")

    @property
    def cross_encoder_available(self) -> bool:
        return _module_available("sentence_transformers")


settings = Settings()
