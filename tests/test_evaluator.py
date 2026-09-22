"""Tests for deterministic evaluation metrics."""

import pytest
from src.core.models import GoldAnswer, RetrievedChunk, BaselineOutput
from src.evaluation.metrics import compute_evidence_metrics, check_claims, evaluate_single_answer


def test_evidence_metrics_exact_match():
    gold = GoldAnswer(
        question_id="Q001",
        policy_family="remote_work",
        gold_answer="Three days currently; four days starts 1 July.",
        required_claims=["three days", "four days", "1 July"],
        forbidden_claims=["four days currently"],
        acceptable_evidence_sets=[["D102#C1", "D103#C1"]],
        question_category="future_effective",
        expected_difficulty="medium"
    )
    chunks = [
        RetrievedChunk(clause_id="D102#C1", document_id="D102", score=0.9, text="...", metadata={}),
        RetrievedChunk(clause_id="D103#C1", document_id="D103", score=0.85, text="...", metadata={}),
        RetrievedChunk(clause_id="D101#C1", document_id="D101", score=0.7, text="...", metadata={})
    ]

    recall, precision = compute_evidence_metrics(chunks, gold)
    assert recall == 1.0
    assert abs(precision - (2/3)) < 1e-4


def test_claims_checking_and_forbidden_violation():
    gold = GoldAnswer(
        question_id="Q001",
        policy_family="remote_work",
        gold_answer="Three days currently; four days starts 1 July.",
        required_claims=["three days", "four days", "1 July"],
        forbidden_claims=["four days currently"],
        acceptable_evidence_sets=[["D102#C1", "D103#C1"]],
        question_category="future_effective",
        expected_difficulty="medium"
    )

    # Compliant answer
    good_text = "Permanent employees are allowed three days per week today, with four days starting on 1 July 2026."
    req_cov, forb_count = check_claims(good_text, gold)
    assert req_cov == 1.0
    assert forb_count == 0

    # Violating answer (outdated reasoning)
    bad_text = "Employees are allowed four days currently under the latest document."
    bad_cov, bad_forb = check_claims(bad_text, gold)
    assert bad_forb == 1


def test_single_answer_eval_unresolved_conflict():
    gold = GoldAnswer(
        question_id="Q011",
        policy_family="support_callback",
        gold_answer="Cannot establish one deadline: sources disagree between four and eight business hours.",
        required_claims=["conflict", "four", "eight"],
        forbidden_claims=["governs four hours definitely"],
        acceptable_evidence_sets=[["D301#C1", "D302#C1"]],
        question_category="unresolved_conflict",
        expected_difficulty="hard"
    )

    correct_conflict_text = "There is an unresolved conflict between sources: one says four hours and another says eight hours."
    chunks = [
        RetrievedChunk(clause_id="D301#C1", document_id="D301", score=0.9, text="...", metadata={}),
        RetrievedChunk(clause_id="D302#C1", document_id="D302", score=0.88, text="...", metadata={})
    ]

    result = evaluate_single_answer(correct_conflict_text, chunks, gold)
    assert result["answer_correctness"] is True
    assert result["conflict_handled_correctly"] is True
