"""
Run 8-question Development Smoke Test for Phase 2 CAKE Pipeline.
Compares performance directly against Phase 1 Conventional RAG Baseline.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.core.models import Question
from src.cake.cake_pipeline import CAKEPipeline
from src.evaluation.evaluator import BaselineEvaluator
from src.evaluation.metrics import compute_cake_metrics


def main():
    parser = argparse.ArgumentParser(description="Run 8-Question Smoke Test for CAKE")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock generator for fast verification without Qwen")
    parser.add_argument("--num-questions", type=int, default=8, help="Number of development questions to run (default: 8)")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. Load config
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["paths"]["data_dir"])
    gt_path = data_dir / "ground_truth" / "development" / "answers.jsonl"
    questions_path = data_dir / "evaluation" / "development" / "questions.jsonl"

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("d:/BTech Project Antigravity/results/cake") / f"smoke_run_{timestamp_str}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2: CONFLICT-AWARE KNOWLEDGE EVOLUTION (CAKE) SMOKE TEST")
    print(f"Embedding: {config['embedding']['model_name']} | LLM: {config['llm']['model_name']} (mock={args.mock_llm})")
    print(f"Results Directory: {results_dir}")
    print("=" * 70)

    # 2. Initialize CAKE pipeline & evaluator
    pipeline = CAKEPipeline(
        data_dir=str(data_dir),
        storage_dir=config["paths"]["storage_dir"],
        embedding_model=config["embedding"]["model_name"],
        llm_model=config["llm"]["model_name"],
        top_k=config["retrieval"]["top_k"],
        mock_generator=args.mock_llm
    )
    print("Preloading models into memory for optimal CPU throughput...")
    pipeline.preload_models()
    evaluator = BaselineEvaluator(ground_truth_path=gt_path)

    # 3. Load smoke questions
    all_questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_questions.append(Question(**json.loads(line)))

    smoke_questions = all_questions[:args.num_questions]
    print(f"Loaded {len(smoke_questions)} development questions for CAKE evaluation.\n")

    # 4. Execute queries through CAKE
    outputs = []
    cake_metrics_list = []
    for idx, q in enumerate(smoke_questions, 1):
        print(f"[{idx}/{len(smoke_questions)}] Processing {q.question_id} (Family: {q.policy_family})...")
        print(f"  Question: {q.question}")
        print(f"  Snapshot: {q.snapshot_at} | Asked At: {q.asked_at}")

        output = pipeline.run_query(q)
        outputs.append(output)

        gold = evaluator.gold_answers[q.question_id]
        cake_met = compute_cake_metrics(output, gold)
        cake_metrics_list.append(cake_met)

        print(f"  Raw Retrieved Chunks    : {[c.clause_id for c in getattr(output, 'raw_retrieved_chunks', [])]}")
        print(f"  CAKE Governing Clause(s): {getattr(output, 'governing_ids', [])}")
        print(f"  CAKE Suppressed Clauses : {getattr(output, 'suppressed_ids', [])}")
        print(f"  CAKE Excluded Future    : {getattr(output, 'future_ids', [])}")
        print(f"  Context Passed to Qwen  : {[c.clause_id for c in output.retrieved_chunks]}")
        print(f"  Generated Answer        : {output.answer}")
        print(f"  Cited Clauses           : {output.cited_clauses}")
        print(f"  Latency                 : {output.latency_ms:.1f} ms\n")

    # 5. Evaluate outputs
    eval_summary = evaluator.evaluate_all(outputs)

    mean_gov_acc = sum(m["governing_clause_accuracy"] for m in cake_metrics_list) / len(cake_metrics_list)
    mean_sup_rate = sum(m["superseded_clause_suppression_rate"] for m in cake_metrics_list) / len(cake_metrics_list)

    # 6. Save results
    answers_file = results_dir / "answers.jsonl"
    with open(answers_file, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(out.model_dump_json() + "\n")

    summary_data = eval_summary.model_dump()
    summary_data["governing_clause_accuracy"] = mean_gov_acc
    summary_data["superseded_clause_suppression_rate"] = mean_sup_rate

    summary_file = results_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 7. Print summary and comparison report
    print("=" * 70)
    print("CAKE SMOKE TEST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Questions Evaluated          : {eval_summary.total_questions}")
    print(f"Answer Accuracy (Exact Claims)     : {eval_summary.mean_accuracy * 100:.1f}%")
    print(f"Required Claim Coverage            : {eval_summary.mean_required_claim_coverage * 100:.1f}%")
    print(f"Forbidden Claim Violation Rate     : {eval_summary.forbidden_claim_violation_rate * 100:.1f}%")
    print(f"Evidence Recall@5                  : {eval_summary.mean_evidence_recall * 100:.1f}%")
    print(f"Evidence Precision@5               : {eval_summary.mean_evidence_precision * 100:.1f}%")
    print(f"Governing Clause Accuracy          : {mean_gov_acc * 100:.1f}%")
    print(f"Superseded Suppression Rate        : {mean_sup_rate * 100:.1f}%")
    print(f"Outdated-Answer Rate               : {eval_summary.outdated_answer_rate * 100:.1f}%")
    print(f"Conflict Handling Rate             : {eval_summary.conflict_handling_rate * 100:.1f}%")
    print(f"Mean Latency                       : {eval_summary.mean_latency_ms:.1f} ms")

    print("\n" + "=" * 70)
    print("PHASE 1 (BASELINE) vs PHASE 2 (CAKE) COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<34} | {'Phase 1 Baseline':<16} | {'Phase 2 CAKE':<14}")
    print("-" * 70)
    print(f"{'Answer Accuracy':<34} | {'25.0%':<16} | {f'{eval_summary.mean_accuracy*100:.1f}%':<14}")
    print(f"{'Evidence Recall@5':<34} | {'100.0%':<16} | {f'{eval_summary.mean_evidence_recall*100:.1f}%':<14}")
    print(f"{'Evidence Precision@5':<34} | {'42.5%':<16} | {f'{eval_summary.mean_evidence_precision*100:.1f}%':<14}")
    print(f"{'Governing Clause Accuracy':<34} | {'37.5%':<16} | {f'{mean_gov_acc*100:.1f}%':<14}")
    print(f"{'Outdated-Answer Rate':<34} | {'25.0%':<16} | {f'{eval_summary.outdated_answer_rate*100:.1f}%':<14}")
    print(f"{'Superseded Suppression Rate':<34} | {'0.0%':<16} | {f'{mean_sup_rate*100:.1f}%':<14}")
    
    q001_eval = evaluator.evaluate_output(outputs[0])
    q001_status = "PASS" if q001_eval.answer_correctness else "FAIL"
    print(f"{'Q001 Outcome':<34} | {'FAIL':<16} | {q001_status:<14}")
    print("=" * 70)
    print(f"CAKE outputs preserved at: {results_dir}\n")

    pipeline.close()


if __name__ == "__main__":
    main()
