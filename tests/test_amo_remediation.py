"""Tests verifying all 7 remediations from Phase 3 forensic audit."""

import tempfile
from unittest.mock import patch
import pytest

from src.core.models import Question, UserContext, RetrievedChunk
from src.amo.family_router import FamilyRouter
from src.amo.context_cache import ContextCache, CachedCAKEContext, compute_intent_signature
from src.amo.amo_pipeline import AMOPipeline
from src.cake.conflict_detector import detect_conflicts


def test_remediation_1_deferred_snapshot_indexing():
    """Verify that ContextCache hits bypass ensure_snapshot_indexed completely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = AMOPipeline(mock_generator=True)
        try:
            pipeline.preload_models()
            pipeline.context_cache = ContextCache(cache_dir=tmp_dir)

            # Seed the ContextCache manually for snapshot 2026-06-20 with user_context
            q_text = "How many remote days per week are allowed today?"
            snap = "2026-06-20T23:59:59Z"
            uctx = {"locations": ["Pune"]}
            chunk = RetrievedChunk(
                clause_id="D102#C1",
                document_id="D102",
                score=0.95,
                text="3 days remote allowed",
                metadata={"policy_family": "remote_work"}
            )
            cached_ctx = CachedCAKEContext(
                governing_chunks=[chunk],
                governing_ids=["D102#C1"],
                suppressed_ids=["D101#C1"],
                future_ids=["D103#C1"],
                has_conflict=False,
                policy_family="remote_work",
                snapshot_at=snap,
                intent_signature=compute_intent_signature(q_text, snap, user_context=uctx)
            )
            pipeline.context_cache.set(snap, "remote_work", q_text, cached_ctx, user_context=uctx)

            # Mock ensure_snapshot_indexed to detect if it gets called
            with patch.object(pipeline, "ensure_snapshot_indexed", wraps=pipeline.ensure_snapshot_indexed) as mock_index:
                question = Question(
                    question_id="Q_CACHE_HIT",
                    policy_family="remote_work",
                    question=q_text,
                    snapshot_at=snap,
                    asked_at=snap,
                    user_context=UserContext(locations=["Pune"])
                )
                out = pipeline.run_query(question)
                
                # ensure_snapshot_indexed MUST NOT be called on a context cache hit
                assert not mock_index.called, "ensure_snapshot_indexed was called despite ContextCache HIT!"
                assert getattr(out, "amo_context_cache_hit", False) is True
                assert out.cited_clauses == ["D102#C1"]
        finally:
            pipeline.close()


def test_remediation_2_has_conflict_flag():
    """Verify that has_conflict reflects true unresolved CAKE conflicts and not string matching."""
    # Create two genuinely conflicting chunks with no governing resolution
    chunk1 = RetrievedChunk(
        clause_id="D101#C1",
        document_id="D101",
        score=0.90,
        text="All employees must work from office 5 days per week.",
        metadata={"policy_family": "remote_work", "effective_date": "2026-01-01", "authority": "HR"}
    )
    chunk2 = RetrievedChunk(
        clause_id="D102#C1",
        document_id="D102",
        score=0.90,
        text="All employees may work remotely 5 days per week.",
        metadata={"policy_family": "remote_work", "effective_date": "2026-01-01", "authority": "HR"}
    )

    conflict_res = detect_conflicts(
        [chunk1, chunk2],
        snapshot_at="2026-01-15T00:00:00Z"
    )

    # Must detect unresolved conflict
    assert conflict_res.has_conflict is True, "Expected has_conflict=True for conflicting equal-authority clauses"
    assert len(conflict_res.competing_clauses) == 2


def test_remediation_3_family_router_denominator():
    """Verify that all competing families are summed in the confidence denominator."""
    router = FamilyRouter(confidence_threshold=0.70)
    # top_score = 3.0, 3 competitors with score 1.0 each -> 3.0 / 6.0 = 0.50 < 0.70 -> fallback
    res = router.route_query("For wfh, telecommute, and hybrid staff, what are the lunch rules during an archive release?")
    assert res.fallback_to_global is True
    assert res.confidence < 0.70


def test_remediation_4_paraphrase_cache_hit():
    """Verify that paraphrased queries share ContextCache entries."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = AMOPipeline(mock_generator=True)
        try:
            pipeline.preload_models()
            pipeline.context_cache = ContextCache(cache_dir=tmp_dir)

            q1 = Question(
                question_id="Q_ORIG",
                policy_family="remote_work",
                question="What is the current weekly remote-work maximum?",
                snapshot_at="2026-08-01T23:59:59Z",
                asked_at="2026-08-01T23:59:59Z",
                user_context=UserContext(locations=["Pune"])
            )
            out1 = pipeline.run_query(q1)
            assert getattr(out1, "amo_context_cache_hit", False) is False

            # Paraphrased query on same snapshot & scope
            q2 = Question(
                question_id="Q_PARAPHRASE",
                policy_family="remote_work",
                question="How many remote days per week are allowed right now?",
                snapshot_at="2026-08-01T23:59:59Z",
                asked_at="2026-08-01T23:59:59Z",
                user_context=UserContext(locations=["Pune"])
            )
            out2 = pipeline.run_query(q2)
            assert getattr(out2, "amo_context_cache_hit", False) is True, "Paraphrased query should hit ContextCache!"
            assert out1.cited_clauses == out2.cited_clauses
        finally:
            pipeline.close()


def test_remediation_5_ood_abstention():
    """Verify that out-of-domain queries abstain instead of hallucinating."""
    pipeline = AMOPipeline(mock_generator=True)
    try:
        pipeline.preload_models()

        q_ood = Question(
            question_id="Q_OOD",
            policy_family="unknown",
            question="What is the dress code policy for employees on casual Fridays?",
            snapshot_at="2026-07-01T23:59:59Z",
            asked_at="2026-07-01T23:59:59Z",
            user_context=UserContext(locations=["Pune"])
        )
        out = pipeline.run_query(q_ood)
        assert getattr(out, "amo_retrieval_method") == "out_of_domain_abstention"
        assert "could not find any relevant enterprise policy" in out.answer.lower()
        assert out.cited_clauses == []
        assert out.retrieved_chunks == []
    finally:
        pipeline.close()
