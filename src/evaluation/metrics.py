"""Deterministic evaluation metrics for enterprise RAG."""

import re
from typing import List, Dict, Any, Tuple
from src.core.models import GoldAnswer, RetrievedChunk, BaselineOutput


def compute_evidence_metrics(
    retrieved_chunks: List[RetrievedChunk],
    gold_answer: GoldAnswer
) -> Tuple[float, float]:
    """
    Computes Evidence Recall and Evidence Precision against acceptable evidence sets.
    If any acceptable evidence set is completely contained in retrieved chunks, recall is 1.0.
    """
    retrieved_ids = set(c.clause_id for c in retrieved_chunks)
    if not retrieved_ids:
        return 0.0, 0.0

    # Best recall across acceptable evidence alternatives
    best_recall = 0.0
    all_acceptable_ids = set()
    for ev_set in gold_answer.acceptable_evidence_sets:
        ev_set_set = set(ev_set)
        all_acceptable_ids.update(ev_set_set)
        intersection = retrieved_ids.intersection(ev_set_set)
        rec = len(intersection) / len(ev_set_set) if ev_set_set else 1.0
        if rec > best_recall:
            best_recall = rec

    # Precision: fraction of retrieved clauses that are in any acceptable set
    relevant_retrieved = retrieved_ids.intersection(all_acceptable_ids)
    precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0

    return best_recall, precision


def check_claims(
    generated_text: str,
    gold_answer: GoldAnswer
) -> Tuple[float, int]:
    """
    Computes:
    - required_claim_coverage: fraction of required claims found in generated answer
    - forbidden_claim_violations: number of forbidden claims found in generated answer
    """
    norm_text = generated_text.lower()
    
    # Required claims
    req_found = 0
    for req in gold_answer.required_claims:
        if req.lower() in norm_text:
            req_found += 1
    
    req_coverage = req_found / len(gold_answer.required_claims) if gold_answer.required_claims else 1.0

    # Forbidden claims
    forb_violations = 0
    for forb in gold_answer.forbidden_claims:
        if forb.lower() in norm_text:
            forb_violations += 1

    return req_coverage, forb_violations


def evaluate_single_answer(
    generated_answer: str,
    retrieved_chunks: List[RetrievedChunk],
    gold: GoldAnswer
) -> Dict[str, Any]:
    """Evaluates a single question answer deterministically."""
    ev_recall, ev_precision = compute_evidence_metrics(retrieved_chunks, gold)
    req_coverage, forb_violations = check_claims(generated_answer, gold)

    # Correctness requires complete required claim coverage and 0 forbidden claims
    is_correct = (req_coverage >= 1.0) and (forb_violations == 0)

    # Outdated answer detection: forbidden claim violation on a current query
    is_outdated = (forb_violations > 0) and (gold.question_category in ["current", "future_effective"])

    # Conflict handling: for unresolved conflict, correctly mentions both sides and uncertainty
    conflict_handled = True
    if gold.question_category == "unresolved_conflict":
        norm = generated_answer.lower()
        has_both = ("four" in norm or "4" in norm) and ("eight" in norm or "8" in norm)
        has_conflict_word = any(w in norm for w in ["conflict", "disagree", "cannot establish", "uncertain", "unresolved"])
        conflict_handled = has_both and has_conflict_word

    return {
        "answer_correctness": is_correct,
        "required_claim_coverage": req_coverage,
        "forbidden_claim_violations": forb_violations,
        "evidence_recall": ev_recall,
        "evidence_precision": ev_precision,
        "outdated_answer_detected": is_outdated,
        "conflict_handled_correctly": conflict_handled
    }


def compute_cake_metrics(
    output: BaselineOutput,
    gold: GoldAnswer
) -> Dict[str, float]:
    """
    Computes CAKE-specific resolution metrics:
    - governing_clause_accuracy: 1.0 if correct governing clause is selected in context
    - superseded_clause_suppression_rate: fraction of superseded versions removed from context
    """
    governing_ids = getattr(output, "governing_ids", [c.clause_id for c in output.retrieved_chunks])
    suppressed_ids = getattr(output, "suppressed_ids", [])
    raw_chunks = getattr(output, "raw_retrieved_chunks", output.retrieved_chunks)

    acceptable_flat = set()
    for s in gold.acceptable_evidence_sets:
        acceptable_flat.update(s)

    # 1. Governing clause accuracy
    governing_acc = 0.0
    if governing_ids:
        if any(gid in acceptable_flat for gid in governing_ids):
            governing_acc = 1.0
    elif gold.question_category == "unresolved_conflict":
        # Unresolved conflict has no single governing winner
        governing_acc = 1.0

    # 2. Superseded suppression rate
    # Count how many chunks in raw_chunks were superseded
    raw_ids = set(c.clause_id for c in raw_chunks)
    passed_ids = set(c.clause_id for c in output.retrieved_chunks)
    
    # If any known obsolete clauses from prior versions were removed
    suppression_rate = 1.0
    if suppressed_ids:
        # All identified superseded clauses were filtered out from final context
        removed_count = sum(1 for sid in suppressed_ids if sid not in passed_ids)
        suppression_rate = removed_count / len(suppressed_ids)

    return {
        "governing_clause_accuracy": governing_acc,
        "superseded_clause_suppression_rate": suppression_rate
    }

