"""RRF fusion correctness."""

from __future__ import annotations

from knowledge_mcp.retrieval.hybrid import rrf_fuse


def test_rrf_rewards_agreement():
    # B is rank-1 in one list and rank-2 in the other; A is rank-1 then absent.
    vector = [("A", 0.9), ("B", 0.8), ("C", 0.1)]
    lexical = [("B", 5.0), ("D", 4.0), ("A", 3.0)]
    fused = dict(rrf_fuse([vector, lexical], k=60))
    # B appears high in both -> should beat A which is high in one, low in the other.
    assert fused["B"] > fused["A"]
    assert set(fused) == {"A", "B", "C", "D"}


def test_rrf_uses_rank_not_score():
    # Identical orderings but wildly different scores must produce identical fusion.
    a = [("X", 1000.0), ("Y", 1.0)]
    b = [("X", 0.001), ("Y", 0.0001)]
    assert rrf_fuse([a]) == rrf_fuse([b])


def test_rrf_top_n_truncates_and_orders():
    lst = [("A", 1), ("B", 1), ("C", 1)]
    out = rrf_fuse([lst], top_n=2)
    assert [mid for mid, _ in out] == ["A", "B"]


def test_rrf_known_value():
    # Single list: score == 1/(k+rank+1). With k=60, rank0 -> 1/61.
    out = dict(rrf_fuse([[("A", 0.0)]], k=60))
    assert abs(out["A"] - 1 / 61) < 1e-12
