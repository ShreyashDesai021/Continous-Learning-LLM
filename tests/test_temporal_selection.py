"""Tests for deterministic temporal rule selection in CAKE."""

import pytest
from src.core.models import RetrievedChunk
from src.cake.temporal_selector import select_governing_clause


def test_q001_temporal_resolution_example():
    """
    Test user specification example:
    snapshot = 2026-06-20
    D101 effective 2025-01-01
    D102 effective 2026-02-01
    D103 effective 2026-07-01
    Result: D102 governs, D101 is superseded, D103 is future-effective.
    """
    family_chunks = [
        RetrievedChunk(
            clause_id="D101#C1",
            document_id="D101",
            score=0.65,
            text="Permanent employees may work remotely for a maximum of two days per week.",
            metadata={"effective_from": "2025-01-01", "approval_status": "approved"}
        ),
        RetrievedChunk(
            clause_id="D102#C1",
            document_id="D102",
            score=0.68,
            text="From 1 February 2026, permanent employees may work remotely for a maximum of three days per week.",
            metadata={"effective_from": "2026-02-01", "approval_status": "approved"}
        ),
        RetrievedChunk(
            clause_id="D103#C1",
            document_id="D103",
            score=0.70,
            text="From 1 July 2026, permanent employees may work remotely for a maximum of four days per week.",
            metadata={"effective_from": "2026-07-01", "approval_status": "approved"}
        ),
    ]

    snapshot_at = "2026-06-20T23:59:59Z"
    governing, superseded, future = select_governing_clause(family_chunks, snapshot_at)

    assert governing is not None
    assert governing.clause_id == "D102#C1"
    
    superseded_ids = [c.clause_id for c in superseded]
    assert "D101#C1" in superseded_ids

    future_ids = [c.clause_id for c in future]
    assert "D103#C1" in future_ids


def test_post_july_snapshot_selects_v3():
    """At 2026-08-01, D103 has become effective and supersedes D102 and D101."""
    family_chunks = [
        RetrievedChunk(
            clause_id="D101#C1",
            document_id="D101",
            score=0.65,
            text="Two days per week",
            metadata={"effective_from": "2025-01-01", "approval_status": "approved"}
        ),
        RetrievedChunk(
            clause_id="D102#C1",
            document_id="D102",
            score=0.68,
            text="Three days per week",
            metadata={"effective_from": "2026-02-01", "approval_status": "approved"}
        ),
        RetrievedChunk(
            clause_id="D103#C1",
            document_id="D103",
            score=0.70,
            text="Four days per week",
            metadata={"effective_from": "2026-07-01", "approval_status": "approved"}
        ),
    ]

    snapshot_at = "2026-08-01T23:59:59Z"
    governing, superseded, future = select_governing_clause(family_chunks, snapshot_at)

    assert governing is not None
    assert governing.clause_id == "D103#C1"
    
    superseded_ids = [c.clause_id for c in superseded]
    assert "D101#C1" in superseded_ids
    assert "D102#C1" in superseded_ids
    assert len(future) == 0


def test_unapproved_proposal_does_not_displace_approved_standard():
    """D402 (unapproved suggestion of 7 days) must not displace D401 (approved 30 days)."""
    family_chunks = [
        RetrievedChunk(
            clause_id="D401#C1",
            document_id="D401",
            score=0.75,
            text="Production application backups must be retained for 30 days.",
            metadata={"effective_from": "2026-01-10", "approval_status": "approved"}
        ),
        RetrievedChunk(
            clause_id="D402#C1",
            document_id="D402",
            score=0.82,
            text="We should retain production application backups for only seven days.",
            metadata={"effective_from": None, "approval_status": "unapproved"}
        )
    ]
    snapshot_at = "2026-03-10T23:59:59Z"
    governing, superseded, future = select_governing_clause(family_chunks, snapshot_at)

    assert governing is not None
    assert governing.clause_id == "D401#C1"
