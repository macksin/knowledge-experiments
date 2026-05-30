# knowledge-mcp

A local-first, production-leaning **MCP server for knowledge & memory**: a bitemporal
graph + Anthropic-style contextual retrieval, organized around a four-type cognitive
taxonomy (semantic / episodic / procedural / self-schema) and a future nightly
"dreaming" consolidation job.

See [`DESIGN.md`](./DESIGN.md) for the full design and phased plan. This repo currently
implements **Phase 0 (scaffold) + Phase 1 (fast-path MVP)**.

## Quickstart

```bash
uv sync                 # install (offline-capable; real model deps are the `real` extra)
uv run pytest -q        # run the test suite
uv run knowledge-mcp    # start the MCP server over stdio
```

The MVP runs **fully offline**: it falls back to a deterministic hash embedder, an
identity reranker, and a no-op contextualizer. Install the real backends with
`uv sync --extra real` and set `ANTHROPIC_API_KEY` to enable bge embeddings,
cross-encoder reranking, and Claude-generated contextual blurbs.

## Tools (Phase 1)

- `remember(content, type?, source?, occurred_at?, links?)` — capture a memory (defaults to `episodic`).
- `recall(query, as_of?, types?, k?)` — hybrid contextual retrieval with optional point-in-time graph context.
- `link(from_id, to_id, relation, valid_from?, valid_to?)` — create a typed bitemporal edge.
- `timeline(entity_or_topic)` — full edge history for an entity.
