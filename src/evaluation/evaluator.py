"""Evaluation runner comparing generated answers to privileged ground truth."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.core.models import GoldAnswer, BaselineOutput, QuestionEvaluation, EvaluationSummary
from src.evaluation.metrics import evaluate_single_answer


class BaselineEvaluator:
    """Loads privileged ground truth answers and scores baseline outputs."""

    def __init__(self, ground_truth_path: Path):
        self.ground_truth_path = Path(ground_truth_path)
        self.gold_answers: Dict[str, GoldAnswer] = self._load_gold_answers()

    def _load_gold_answers(self) -> Dict[str, GoldAnswer]:
        gold_map = {}
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                gold_map[item["question_id"]] = GoldAnswer(**item)
        return gold_map

    def evaluate_output(self, output: BaselineOutput) -> QuestionEvaluation:
        """Evaluates a single BaselineOutput against gold."""
        gold = self.gold_answers[output.question_id]
        eval_dict = evaluate_single_answer(
            generated_answer=output.answer,
            retrieved_chunks=output.retrieved_chunks,
            gold=gold
        )
        return QuestionEvaluation(
            question_id=output.question_id,
            policy_family=gold.policy_family,
            question_category=gold.question_category,
            answer_correctness=eval_dict["answer_correctness"],
            required_claim_coverage=eval_dict["required_claim_coverage"],
            forbidden_claim_violations=eval_dict["forbidden_claim_violations"],
            evidence_recall=eval_dict["evidence_recall"],
            evidence_precision=eval_dict["evidence_precision"],
            outdated_answer_detected=eval_dict["outdated_answer_detected"],
            conflict_handled_correctly=eval_dict["conflict_handled_correctly"],
            latency_ms=output.latency_ms,
            generated_answer=output.answer,
            gold_answer=gold.gold_answer,
            retrieved_clauses=[c.clause_id for c in output.retrieved_chunks]
        )

    def evaluate_all(self, outputs: List[BaselineOutput]) -> EvaluationSummary:
        """Evaluates all outputs and generates category breakdown."""
        evals = [self.evaluate_output(out) for out in outputs]
        n = len(evals)
        if n == 0:
            raise ValueError("No outputs to evaluate.")

        mean_acc = sum(1 for e in evals if e.answer_correctness) / n
        mean_req = sum(e.required_claim_coverage for e in evals) / n
        forb_rate = sum(1 for e in evals if e.forbidden_claim_violations > 0) / n
        mean_rec = sum(e.evidence_recall for e in evals) / n
        mean_prec = sum(e.evidence_precision for e in evals) / n
        
        # Outdated rate on eligible questions (current / future_effective)
        current_evals = [e for e in evals if e.question_category in ["current", "future_effective"]]
        outdated_rate = (
            sum(1 for e in current_evals if e.outdated_answer_detected) / len(current_evals)
            if current_evals else 0.0
        )

        # Conflict handling on conflict questions
        conflict_evals = [e for e in evals if e.question_category == "unresolved_conflict"]
        conflict_rate = (
            sum(1 for e in conflict_evals if e.conflict_handled_correctly) / len(conflict_evals)
            if conflict_evals else 1.0
        )

        mean_lat = sum(e.latency_ms for e in evals) / n

        # Category breakdown
        cat_map: Dict[str, List[QuestionEvaluation]] = {}
        for e in evals:
            cat_map.setdefault(e.question_category, []).append(e)

        breakdown = {}
        for cat, items in cat_map.items():
            k = len(items)
            breakdown[cat] = {
                "count": k,
                "accuracy": sum(1 for x in items if x.answer_correctness) / k,
                "evidence_recall": sum(x.evidence_recall for x in items) / k,
                "evidence_precision": sum(x.evidence_precision for x in items) / k
            }

        return EvaluationSummary(
            total_questions=n,
            mean_accuracy=mean_acc,
            mean_required_claim_coverage=mean_req,
            forbidden_claim_violation_rate=forb_rate,
            mean_evidence_recall=mean_rec,
            mean_evidence_precision=mean_prec,
            outdated_answer_rate=outdated_rate,
            conflict_handling_rate=conflict_rate,
            mean_latency_ms=mean_lat,
            category_breakdown=breakdown
        )
