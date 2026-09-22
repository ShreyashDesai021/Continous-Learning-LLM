"""Tests for conflict detection in CAKE."""

import pytest
from src.core.models import RetrievedChunk
from src.cake.conflict_detector import detect_conflicts


def test_unresolved_callback_conflict_detected():
    """Support callback guide (4h) and onboarding note (8h) must trigger conflict."""
    family_chunks = [
        RetrievedChunk(
            clause_id="D301#C1",
            document_id="D301",
            score=0.88,
            text="support team must call the customer within four business hours.",
            metadata={"effective_from": None, "approval_status": "unknown"}
        ),
        RetrievedChunk(
            clause_id="D302#C1",
            document_id="D302",
            score=0.85,
            text="support team must call the customer within eight business hours.",
            metadata={"effective_from": None, "approval_status": "unknown"}
        )
    ]
    snapshot_at = "2026-04-10T23:59:59Z"
    res = detect_conflicts(family_chunks, snapshot_at)

    assert res.has_conflict is True
    assert len(res.competing_clauses) == 2
    competing_ids = [c.clause_id for c in res.competing_clauses]
    assert "D301#C1" in competing_ids
    assert "D302#C1" in competing_ids
    assert res.governing_clause is None


def test_resolved_supersession_has_no_conflict():
    """When a newer approved policy supersedes an older one, conflict is resolved (has_conflict=False)."""
    family_chunks = [
        RetrievedChunk(
            clause_id="D101#C1",
            document_id="D101",
            score=0.65,
            text="maximum of two days per week.",
            metadata={"effective_from": "2025-01-01", "approval_status": "approved"}
        ),
        RetrievedChunk(
            clause_id="D102#C1",
            document_id="D102",
            score=0.68,
            text="maximum of three days per week.",
            metadata={"effective_from": "2026-02-01", "approval_status": "approved"}
        )
    ]
    snapshot_at = "2026-02-10T23:59:59Z"
    res = detect_conflicts(family_chunks, snapshot_at)

    assert res.has_conflict is False
    assert res.governing_clause is not None
    assert res.governing_clause.clause_id == "D102#C1"
    assert len(res.superseded_clauses) == 1
    assert res.superseded_clauses[0].clause_id == "D101#C1"
