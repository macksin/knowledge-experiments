"""Consolidator provider: the reasoning step of the dreaming phase.

This is where "things learned along the way" stop being one undifferentiated
stream. A consolidator reads a raw *episodic* memory and proposes one or more
decontextualized memories of a more durable type:

    self_schema -- a preference / value / disposition about the user
    procedural  -- a reusable how-to / heuristic ("when X, do Z")
    semantic    -- a decontextualized fact about the world

`HeuristicConsolidator` (offline default) classifies with transparent regex cues so
the whole pipeline runs deterministically with no network. `AnthropicConsolidator`
does the real reasoning with Claude when a key is set. Both share the seam, exactly
like the embedder / reranker / contextualizer providers.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from knowledge_mcp.config import settings
from knowledge_mcp.models import Memory, MemoryType

# First-person + evaluative cues -> the memory is about the *user* (self_schema).
_FIRST_PERSON = re.compile(r"\b(i|i'm|i am|my|me|mine)\b", re.IGNORECASE)
_EVALUATIVE = re.compile(
    r"\b(prefer|favou?rite|like|love|hate|dislike|value|believe|tend to|usually|always)\b",
    re.IGNORECASE,
)
# How-to / conditional cues -> procedural.
_PROCEDURAL = re.compile(
    r"(\bwhen\b.+\b(then|do|use|run|try|should)\b)"
    r"|(\bif\b.+\bthen\b)"
    r"|(^to\s+\w+)"
    r"|(\b(step|steps|first.+then|the way to)\b)",
    re.IGNORECASE,
)


class ConsolidationProposal(BaseModel):
    """One durable memory a consolidator proposes from a source episode."""

    content: str
    type: MemoryType
    confidence: float = 0.7


@runtime_checkable
class Consolidator(Protocol):
    def consolidate(self, episode: Memory) -> list[ConsolidationProposal]: ...


class HeuristicConsolidator:
    """Deterministic, network-free classifier (offline default).

    Not semantically deep, but transparent and stable: it routes each episode to
    self_schema / procedural / semantic by surface cues so the dreaming pipeline,
    its provenance links, and its tests all run with no model.
    """

    def classify(self, text: str) -> MemoryType:
        if _FIRST_PERSON.search(text) and _EVALUATIVE.search(text):
            return MemoryType.SELF_SCHEMA
        if _PROCEDURAL.search(text):
            return MemoryType.PROCEDURAL
        return MemoryType.SEMANTIC

    def consolidate(self, episode: Memory) -> list[ConsolidationProposal]:
        text = episode.content.strip()
        if not text:
            return []
        return [ConsolidationProposal(content=text, type=self.classify(text))]


class AnthropicConsolidator:
    """Real consolidation with Claude: extract durable facts/strategies/preferences
    from a raw episode and tag each with its memory type. Lazily initialized."""

    _PROMPT = (
        "You are the consolidation ('dreaming') step of a memory system. Read a raw "
        "episodic memory and extract durable knowledge from it. For each item output a "
        "JSON object {\"content\": str, \"type\": one of "
        "[\"semantic\", \"procedural\", \"self_schema\"], \"confidence\": 0..1}. "
        "semantic = a decontextualized fact about the world; procedural = a reusable "
        "how-to or heuristic; self_schema = a preference/value/disposition about the "
        "user. Return a JSON array (possibly empty). Output only the JSON."
    )

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # lazy optional import

        self._client = anthropic.Anthropic()
        self._model = model or settings.anthropic_model

    def consolidate(self, episode: Memory) -> list[ConsolidationProposal]:
        import json

        msg = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=[
                {"type": "text", "text": self._PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": f"EPISODE:\n{episode.content}"}],
        )
        try:
            items = json.loads(msg.content[0].text)
        except (json.JSONDecodeError, IndexError):
            return []
        proposals: list[ConsolidationProposal] = []
        for it in items:
            try:
                proposals.append(
                    ConsolidationProposal(
                        content=it["content"],
                        type=MemoryType(it["type"]),
                        confidence=float(it.get("confidence", 0.7)),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        return proposals


def default_consolidator() -> Consolidator:
    if settings.anthropic_available:
        try:
            return AnthropicConsolidator()
        except Exception:
            pass
    return HeuristicConsolidator()
