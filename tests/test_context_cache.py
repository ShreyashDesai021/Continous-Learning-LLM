"""Unit tests for ContextCache: intent discrimination, snapshot isolation, and persistence."""

import tempfile
import pytest
from src.core.models import RetrievedChunk
from src.amo.context_cache import (
    ContextCache,
    CachedCAKEContext,
    compute_intent_signature
)


def _make_dummy_chunk(clause_id: str, doc_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        clause_id=clause_id,
        document_id=doc_id,
        score=0.95,
        text=text,
        metadata={"policy_family": "remote_work", "locations": ["Pune", "Mumbai"]}
    )


def test_intent_signature_differentiation():
    snapshot = "2026-08-01T23:59:59Z"

    # 1. Historical query vs Current query
    sig_hist = compute_intent_signature("What was the remote-work allowance throughout 2025?", snapshot)
    sig_curr = compute_intent_signature("What is the current weekly remote-work maximum?", snapshot)
    assert sig_hist != sig_curr, "Historical query must produce distinct signature from current query"

    # 2. Sequence history query vs Standard lookup
    sig_seq = compute_intent_signature("Give the sequence of remote-work allowances since January 2025.", snapshot)
    assert sig_seq != sig_curr, "Sequence history query must produce distinct signature from standard query"
    assert sig_seq != sig_hist

    # 3. Geographic scope: Pune vs Mumbai
    sig_pune = compute_intent_signature(
        "What is the meal ceiling for late work?",
        "2026-07-01T23:59:59Z",
        user_context={"locations": ["Pune"], "conditions": []}
    )
    sig_mumbai = compute_intent_signature(
        "What is the meal ceiling for late work?",
        "2026-07-01T23:59:59Z",
        user_context={"locations": ["Mumbai"], "conditions": []}
    )
    assert sig_pune != sig_mumbai, "Pune scope must produce distinct signature from Mumbai scope"


def test_context_cache_snapshot_isolation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ContextCache(cache_dir=tmp_dir, max_entries_per_family=50)

        chunk_v2 = _make_dummy_chunk("D102#C1", "D102", "3 days remote allowed")
        chunk_v3 = _make_dummy_chunk("D103#C1", "D103", "4 days remote allowed")

        ctx_snap1 = CachedCAKEContext(
            governing_chunks=[chunk_v2],
            governing_ids=["D102#C1"],
            suppressed_ids=["D101#C1"],
            future_ids=["D103#C1"],
            has_conflict=False,
            policy_family="remote_work",
            snapshot_at="2026-06-20T23:59:59Z",
            intent_signature="sig_test_1"
        )

        ctx_snap2 = CachedCAKEContext(
            governing_chunks=[chunk_v3],
            governing_ids=["D103#C1"],
            suppressed_ids=["D101#C1", "D102#C1"],
            future_ids=[],
            has_conflict=False,
            policy_family="remote_work",
            snapshot_at="2026-08-01T23:59:59Z",
            intent_signature="sig_test_2"
        )

        query = "How many remote days are allowed today?"
        cache.set("2026-06-20T23:59:59Z", "remote_work", query, ctx_snap1)
        cache.set("2026-08-01T23:59:59Z", "remote_work", query, ctx_snap2)

        # Retrieve at snapshot 1
        res1 = cache.get("2026-06-20T23:59:59Z", "remote_work", query)
        assert res1 is not None
        assert res1.governing_ids == ["D102#C1"]
        assert "D103#C1" in res1.future_ids

        # Retrieve at snapshot 2
        res2 = cache.get("2026-08-01T23:59:59Z", "remote_work", query)
        assert res2 is not None
        assert res2.governing_ids == ["D103#C1"]

        # Cross-snapshot isolation: querying with non-existent snapshot date must miss
        res_other = cache.get("2025-01-01T00:00:00Z", "remote_work", query)
        assert res_other is None


def test_context_cache_disk_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache1 = ContextCache(cache_dir=tmp_dir, max_entries_per_family=50)
        chunk = _make_dummy_chunk("D203#C1", "D203", "Pune meal ceiling INR 250")
        ctx = CachedCAKEContext(
            governing_chunks=[chunk],
            governing_ids=["D203#C1"],
            suppressed_ids=[],
            future_ids=[],
            has_conflict=False,
            policy_family="office_meals",
            snapshot_at="2026-07-01T23:59:59Z",
            intent_signature="sig_pune_meal"
        )

        query = "What is the meal ceiling for Pune?"
        uctx = {"locations": ["Pune"], "conditions": []}
        cache1.set("2026-07-01T23:59:59Z", "office_meals", query, ctx, user_context=uctx)

        # Cold restart
        cache2 = ContextCache(cache_dir=tmp_dir, max_entries_per_family=50)
        assert len(cache2._cache) == 0

        loaded = cache2.get("2026-07-01T23:59:59Z", "office_meals", query, user_context=uctx)
        assert loaded is not None
        assert loaded.governing_ids == ["D203#C1"]
        assert loaded.governing_chunks[0].text == "Pune meal ceiling INR 250"


def test_intent_signature_paraphrase_equivalence():
    """Paraphrased queries within the same family, snapshot, and topic must produce identical intent signatures."""
    snapshot = "2026-08-01T23:59:59Z"
    uctx = {"locations": ["Pune", "Mumbai"], "conditions": []}

    sig1 = compute_intent_signature("What is the current weekly remote-work maximum?", snapshot, user_context=uctx)
    sig2 = compute_intent_signature("How many remote days per week are allowed right now?", snapshot, user_context=uctx)
    assert sig1 == sig2, f"Paraphrases must yield identical intent signature: {sig1} vs {sig2}"


def test_context_cache_atomic_write_safety():
    """Verify that ContextCache writes atomically without leaving corrupt temp files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ContextCache(cache_dir=tmp_dir)
        chunk = _make_dummy_chunk("D101#C1", "D101", "2 days remote allowed")
        ctx = CachedCAKEContext(
            governing_chunks=[chunk],
            governing_ids=["D101#C1"],
            suppressed_ids=[],
            future_ids=[],
            has_conflict=False,
            policy_family="remote_work",
            snapshot_at="2026-06-20T23:59:59Z",
            intent_signature="sig_test"
        )
        cache.set("2026-06-20T23:59:59Z", "remote_work", "What is the remote work policy?", ctx)

        # Ensure no dangling .tmp files exist
        tmp_files = list(cache.cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Found dangling temporary files: {tmp_files}"
        
        # Ensure json file exists and is valid
        json_files = list(cache.cache_dir.glob("*.json"))
        assert len(json_files) == 1
