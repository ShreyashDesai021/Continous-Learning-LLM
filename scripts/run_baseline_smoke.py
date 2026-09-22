"""
Run 8-question Development Smoke Test for Conventional RAG Baseline.
Outputs per-question retrieval, answers, and deterministic evaluation scores.
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
from src.baseline.pipeline import ConventionalRAGPipeline
from src.evaluation.evaluator import BaselineEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run 8-Question Smoke Test for Conventional RAG")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock generator for instant validation without loading Qwen")
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
    results_dir = Path(config["paths"]["results_dir"]) / f"smoke_run_{timestamp_str}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENTERPRISE CLLM: CONVENTIONAL RAG BASELINE SMOKE TEST")
    print(f"Embedding: {config['embedding']['model_name']} | LLM: {config['llm']['model_name']} (mock={args.mock_llm})")
    print(f"Results Directory: {results_dir}")
    print("=" * 70)

    # 2. Initialize pipeline & evaluator
    pipeline = ConventionalRAGPipeline(
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
    print(f"Loaded {len(smoke_questions)} development questions for smoke testing.\n")

    # 4. Execute queries
    outputs = []
    for idx, q in enumerate(smoke_questions, 1):
        print(f"[{idx}/{len(smoke_questions)}] Processing {q.question_id} (Family: {q.policy_family})...")
        print(f"  Question: {q.question}")
        print(f"  Snapshot: {q.snapshot_at} | Asked At: {q.asked_at}")

        output = pipeline.run_query(q)
        outputs.append(output)

        print(f"  Retrieved Chunks (Top-{len(output.retrieved_chunks)}):")
        for chunk in output.retrieved_chunks:
            print(f"    - [{chunk.score:.4f}] {chunk.clause_id} ({chunk.metadata.get('approval_status')}, Eff: {chunk.metadata.get('effective_from')})")
        print(f"  Generated Answer: {output.answer}")
        print(f"  Cited Clauses: {output.cited_clauses}")
        print(f"  Latency: {output.latency_ms:.1f} ms\n")

    # 5. Evaluate outputs
    eval_summary = evaluator.evaluate_all(outputs)

    # 6. Save results
    answers_file = results_dir / "answers.jsonl"
    with open(answers_file, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(out.model_dump_json() + "\n")

    summary_file = results_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(eval_summary.model_dump_json(indent=2) + "\n")

    # 7. Print summary report
    print("=" * 70)
    print("SMOKE TEST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Questions Evaluated       : {eval_summary.total_questions}")
    print(f"Answer Accuracy (Exact Claims)  : {eval_summary.mean_accuracy * 100:.1f}%")
    print(f"Required Claim Coverage         : {eval_summary.mean_required_claim_coverage * 100:.1f}%")
    print(f"Forbidden Claim Violation Rate  : {eval_summary.forbidden_claim_violation_rate * 100:.1f}%")
    print(f"Evidence Recall@5               : {eval_summary.mean_evidence_recall * 100:.1f}%")
    print(f"Evidence Precision@5            : {eval_summary.mean_evidence_precision * 100:.1f}%")
    print(f"Outdated-Answer Rate            : {eval_summary.outdated_answer_rate * 100:.1f}%")
    print(f"Conflict Handling Rate          : {eval_summary.conflict_handling_rate * 100:.1f}%")
    print(f"Mean Latency                    : {eval_summary.mean_latency_ms:.1f} ms")
    print("\nCategory Breakdown:")
    for cat, stats in eval_summary.category_breakdown.items():
        print(f"  - {cat:22s} [N={stats['count']}]: Acc={stats['accuracy']*100:5.1f}% | EvRec={stats['evidence_recall']*100:5.1f}%")
    print("=" * 70)
    print(f"Outputs and metrics successfully preserved at:\n{results_dir}")
    pipeline.close()


if __name__ == "__main__":
    main()
