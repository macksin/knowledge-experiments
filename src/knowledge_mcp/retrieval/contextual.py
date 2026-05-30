"""Contextualizer provider: situates a memory before indexing.

This is Anthropic's Contextual Retrieval adapted to memories: instead of "the
surrounding document", the context for a memory is its **graph neighbors** (parent
project, source thread, linked entities). A short blurb is prepended to the content
before it is embedded and FTS-indexed, which sharply reduces retrieval failures.

`NoOpContextualizer` (offline default) returns an empty blurb.
`AnthropicContextualizer` generates one with Claude + prompt caching when a key is set.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from knowledge_mcp.config import settings
from knowledge_mcp.models import Edge, Memory


@runtime_checkable
class Contextualizer(Protocol):
    def contextualize(self, memory: Memory, neighbors: list[tuple[Edge, Memory]]) -> str: ...


class NoOpContextualizer:
    def contextualize(self, memory: Memory, neighbors: list[tuple[Edge, Memory]]) -> str:
        return ""


class AnthropicContextualizer:
    """Generate a 1-2 sentence situating blurb. Prompt caching keeps cost ~$1/M tok."""

    _PROMPT = (
        "You situate a memory within its graph context so it can be retrieved "
        "accurately later. Given the memory and its linked neighbors, write a short "
        "(1-2 sentence) blurb that names the project/entities/timeframe it belongs to. "
        "Output only the blurb."
    )

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # lazy optional import

        self._client = anthropic.Anthropic()
        self._model = model or settings.anthropic_model

    def contextualize(self, memory: Memory, neighbors: list[tuple[Edge, Memory]]) -> str:
        neighbor_lines = "\n".join(
            f"- [{e.relation}] {m.content}" for e, m in neighbors
        ) or "(no linked neighbors)"
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=150,
            system=[{"type": "text", "text": self._PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[
                {
                    "role": "user",
                    "content": f"MEMORY:\n{memory.content}\n\nNEIGHBORS:\n{neighbor_lines}",
                }
            ],
        )
        return msg.content[0].text.strip()


def default_contextualizer() -> Contextualizer:
    if settings.anthropic_available:
        try:
            return AnthropicContextualizer()
        except Exception:
            pass
    return NoOpContextualizer()
