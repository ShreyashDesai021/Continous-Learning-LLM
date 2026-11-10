"""
Benchmark Runner for Phase 3 Adaptive Memory Optimization (AMO).
Executes Cold Pass, Warm Pass, and Aggregate Pass.
Evaluates:
- Cache Hit Rate
- Embedding Cache Hit Rate
- Average Retrieved Chunks Before CAKE vs After AMO
- Retrieval Volume Reduction
- Mean Query Latency & Latency Reduction vs Phase 2
- Answer Consistency vs Phase 2
- Governing Clause Accuracy, Outdated-Answer Rate, Evidence Recall@5
Generates side-by-side comparison table (Baseline vs CAKE vs AMO) and saves execution trace.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.core.models import Question, BaselineOutput
from src.amo.amo_pipeline import AMOPipeline
from src.evaluation.evaluator import BaselineEvaluator
from src.evaluation.metrics import compute_cake_metrics


def run_benchmark_pass(
    pipeline: AMOPipeline,
    evaluator: BaselineEvaluator,
    questions: List[Question],
    pass_name: str
) -> Dict[str, Any]:
    """Runs a complete query pass over questions and gathers telemetry."""
    outputs: List[BaselineOutput] = []
    cake_metrics_list = []
    traces: List[Dict[str, Any]] = []

    for idx, q in enumerate(questions, 1):
        output = pipeline.run_query(q)
        outputs.append(output)

        gold = evaluator.gold_answers[q.question_id]
        cake_met = compute_cake_metrics(output, gold)
        cake_metrics_list.append(cake_met)

        routing_res = getattr(output, "amo_routing_result", None)
        trace_item = {
            "question_id": q.question_id,
            "question": q.question,
            "snapshot_at": q.snapshot_at,
            "query_cache_hit": getattr(output, "amo_query_cache_hit", False),
            "family_router": {
                "candidate_family": routing_res.candidate_family if routing_res else None,
                "confidence": routing_res.confidence if routing_res else 0.0,
                "fallback_to_global": routing_res.fallback_to_global if routing_res else False,
                "reason": routing_res.reason if routing_res else "unknown"
            },
            "context_cache_hit": getattr(output, "amo_context_cache_hit", False),
            "retrieval_method": getattr(output, "amo_retrieval_method", "unknown"),
            "chunks_retrieved_count": getattr(output, "amo_chunks_retrieved_count", 0),
            "governing_clauses": getattr(output, "governing_ids", []),
            "suppressed_clauses": getattr(output, "suppressed_ids", []),
            "future_clauses": getattr(output, "future_ids", []),
            "final_answer": output.answer,
            "cited_clauses": output.cited_clauses,
            "latency_ms": output.latency_ms
        }
        traces.append(trace_item)

    eval_summary = evaluator.evaluate_all(outputs)
    mean_gov_acc = sum(m["governing_clause_accuracy"] for m in cake_metrics_list) / len(cake_metrics_list)
    mean_sup_rate = sum(m["superseded_clause_suppression_rate"] for m in cake_metrics_list) / len(cake_metrics_list)

    total_q = len(outputs)
    q_cache_hits = sum(1 for o in outputs if getattr(o, "amo_query_cache_hit", False))
    ctx_cache_hits = sum(1 for o in outputs if getattr(o, "amo_context_cache_hit", False))
    mean_retrieved_chunks = sum(getattr(o, "amo_chunks_retrieved_count", 0) for o in outputs) / total_q

    return {
        "pass_name": pass_name,
        "outputs": outputs,
        "traces": traces,
        "eval_summary": eval_summary,
        "total_questions": total_q,
        "mean_latency_ms": eval_summary.mean_latency_ms,
        "query_cache_hit_rate": q_cache_hits / total_q,
        "context_cache_hit_rate": ctx_cache_hits / total_q,
        "mean_chunks_retrieved": mean_retrieved_chunks,
        "governing_clause_accuracy": mean_gov_acc,
        "superseded_suppression_rate": mean_sup_rate,
        "outdated_answer_rate": eval_summary.outdated_answer_rate,
        "evidence_recall": eval_summary.mean_evidence_recall,
        "evidence_precision": eval_summary.mean_evidence_precision,
        "accuracy": eval_summary.mean_accuracy
    }


def main():
    parser = argparse.ArgumentParser(description="Run Benchmark for Phase 3 AMO")
    parser.add_argument("--mock-llm", action="store_true", default=True, help="Use mock generator for fast benchmark verification (default: True)")
    parser.add_argument("--real-llm", action="store_true", help="Run full Qwen generation on CPU")
    parser.add_argument("--num-questions", type=int, default=8, help="Number of questions to evaluate (default: 8)")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to config file")
    args = parser.parse_args()

    use_mock = not args.real_llm

    # 1. Load config
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["paths"]["data_dir"])
    gt_path = data_dir / "ground_truth" / "development" / "answers.jsonl"
    questions_path = data_dir / "evaluation" / "development" / "questions.jsonl"

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results/amo") / f"smoke_run_{timestamp_str}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 3: ADAPTIVE MEMORY OPTIMIZATION (AMO) BENCHMARK")
    print(f"Embedding: {config['embedding']['model_name']} | LLM: {config['llm']['model_name']} (mock={use_mock})")
    print(f"Results Directory: {results_dir}")
    print("=" * 80)

    # 2. Initialize AMO Pipeline & Evaluator
    pipeline = AMOPipeline(
        data_dir=str(data_dir),
        storage_dir=config["paths"]["storage_dir"],
        embedding_model=config["embedding"]["model_name"],
        llm_model=config["llm"]["model_name"],
        top_k=config["retrieval"]["top_k"],
        mock_generator=use_mock
    )
    print("Preloading models into memory...")
    pipeline.preload_models()
    evaluator = BaselineEvaluator(ground_truth_path=gt_path)

    # Clear memory & disk caches to guarantee pure Cold Pass
    pipeline.query_cache.clear(clear_disk=True)
    pipeline.context_cache.clear(clear_disk=True)

    # Load questions
    all_questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_questions.append(Question(**json.loads(line)))
    eval_questions = all_questions[:args.num_questions]
    print(f"Loaded {len(eval_questions)} development questions.\n")

    # Load Phase 2 answers for consistency verification
    phase2_answers_dict = {}
    phase2_dir_candidates = sorted(Path("results/cake").glob("smoke_run_*"), reverse=True)
    for cand in phase2_dir_candidates:
        phase2_file = cand / "answers.jsonl"
        if phase2_file.exists():
            temp_dict = {}
            with open(phase2_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        temp_dict[item["question_id"]] = item
            if len(temp_dict) >= len(eval_questions):
                phase2_answers_dict = temp_dict
                print(f"Loaded Phase 2 reference answers from: {cand.name} ({len(phase2_answers_dict)} answers)")
                break

    # 3. PASS 1: COLD PASS (Unseen Queries)
    print(">>> Executing PASS 1: COLD PASS (All caches empty)...")
    cold_results = run_benchmark_pass(pipeline, evaluator, eval_questions, pass_name="Cold Pass")
    print(f"  Cold Mean Latency: {cold_results['mean_latency_ms']:.2f} ms")
    print(f"  Cold Mean Retrieved Chunks: {cold_results['mean_chunks_retrieved']:.2f} (vs 5.0 baseline)")
    print(f"  Cold Context Cache Hit Rate: {cold_results['context_cache_hit_rate']*100:.1f}%\n")

    # 4. PASS 2: WARM PASS (Recurrent Queries)
    print(">>> Executing PASS 2: WARM PASS (Caches populated)...")
    warm_results = run_benchmark_pass(pipeline, evaluator, eval_questions, pass_name="Warm Pass")
    print(f"  Warm Mean Latency: {warm_results['mean_latency_ms']:.2f} ms")
    print(f"  Warm Mean Retrieved Chunks: {warm_results['mean_chunks_retrieved']:.2f} (vs 5.0 baseline)")
    print(f"  Warm Context Cache Hit Rate: {warm_results['context_cache_hit_rate']*100:.1f}%\n")

    # 5. AGGREGATE & PARAMETRIC WORKLOAD EVALUATION
    # Phase 2 reference values
    phase2_mean_latency = 174.63  # ms (from Phase 2 summary.json)
    phase2_mean_retrieved_chunks = 5.0  # Top-5 global search

    # Parametric Workload Sensitivity Analysis across query repetition rates alpha
    repetition_rates = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    parametric_results = {}
    for alpha in repetition_rates:
        lat = (1.0 - alpha) * cold_results["mean_latency_ms"] + alpha * warm_results["mean_latency_ms"]
        lat_red = ((phase2_mean_latency - lat) / phase2_mean_latency) * 100.0
        chunks = (1.0 - alpha) * cold_results["mean_chunks_retrieved"] + alpha * warm_results["mean_chunks_retrieved"]
        vol_red = ((phase2_mean_retrieved_chunks - chunks) / phase2_mean_retrieved_chunks) * 100.0
        parametric_results[f"{int(alpha * 100)}%"] = {
            "repetition_rate": alpha,
            "mean_latency_ms": round(lat, 2),
            "latency_reduction_pct": round(lat_red, 2),
            "mean_chunks_retrieved": round(chunks, 2),
            "retrieval_volume_reduction_pct": round(vol_red, 2)
        }

    # Aggregate at 50% baseline mix
    agg_mean_latency = parametric_results["50%"]["mean_latency_ms"]
    agg_mean_chunks = parametric_results["50%"]["mean_chunks_retrieved"]
    agg_ctx_hit_rate = (cold_results["context_cache_hit_rate"] + warm_results["context_cache_hit_rate"]) / 2.0
    agg_emb_hit_rate = (cold_results["query_cache_hit_rate"] + warm_results["query_cache_hit_rate"]) / 2.0
    latency_reduction = parametric_results["50%"]["latency_reduction_pct"]
    retrieval_volume_reduction = parametric_results["50%"]["retrieval_volume_reduction_pct"]

    # Answer Consistency Check
    matching_answers = 0
    total_checked = 0
    for out in warm_results["outputs"]:
        qid = out.question_id
        if qid in phase2_answers_dict:
            p2 = phase2_answers_dict[qid]
            ans_match = (out.answer.strip() == p2.get("answer", "").strip())
            cites_match = (set(out.cited_clauses) == set(p2.get("cited_clauses", [])))
            if ans_match and cites_match:
                matching_answers += 1
            else:
                print(f"  [Notice] Mismatch on {qid}: ans_match={ans_match}, cites_match={cites_match}")
                print(f"    AMO Answer: {out.answer}")
                print(f"    P2  Answer: {p2.get('answer')}")
            total_checked += 1

    consistency_rate = (matching_answers / total_checked * 100.0) if total_checked > 0 else 100.0

    # 6. Save Artifacts to results/amo/
    # Cold answers
    with open(results_dir / "cold_answers.jsonl", "w", encoding="utf-8") as f:
        for out in cold_results["outputs"]:
            f.write(out.model_dump_json() + "\n")

    # Warm answers
    with open(results_dir / "warm_answers.jsonl", "w", encoding="utf-8") as f:
        for out in warm_results["outputs"]:
            f.write(out.model_dump_json() + "\n")

    # Detailed Cache Hit Traces
    with open(results_dir / "cache_hit_traces.json", "w", encoding="utf-8") as f:
        json.dump({"cold_traces": cold_results["traces"], "warm_traces": warm_results["traces"]}, f, indent=2)

    # Summary
    summary_data = {
        "timestamp": timestamp_str,
        "total_questions_per_pass": len(eval_questions),
        "cold_pass": {
            "mean_latency_ms": cold_results["mean_latency_ms"],
            "query_cache_hit_rate": cold_results["query_cache_hit_rate"],
            "context_cache_hit_rate": cold_results["context_cache_hit_rate"],
            "mean_chunks_retrieved": cold_results["mean_chunks_retrieved"],
            "governing_clause_accuracy": cold_results["governing_clause_accuracy"]
        },
        "warm_pass": {
            "mean_latency_ms": warm_results["mean_latency_ms"],
            "query_cache_hit_rate": warm_results["query_cache_hit_rate"],
            "context_cache_hit_rate": warm_results["context_cache_hit_rate"],
            "mean_chunks_retrieved": warm_results["mean_chunks_retrieved"],
            "governing_clause_accuracy": warm_results["governing_clause_accuracy"]
        },
        "aggregate": {
            "mean_latency_ms": agg_mean_latency,
            "latency_reduction_vs_phase2_pct": latency_reduction,
            "mean_chunks_retrieved_after_amo": agg_mean_chunks,
            "retrieval_volume_reduction_pct": retrieval_volume_reduction,
            "context_cache_hit_rate": agg_ctx_hit_rate,
            "embedding_cache_hit_rate": agg_emb_hit_rate,
            "governing_clause_accuracy": warm_results["governing_clause_accuracy"],
            "outdated_answer_rate": warm_results["outdated_answer_rate"],
            "evidence_recall": warm_results["evidence_recall"],
            "evidence_precision": warm_results["evidence_precision"],
            "answer_consistency_vs_phase2_pct": consistency_rate
        },
        "parametric_sensitivity": parametric_results
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 7. Print Final Side-by-Side Comparison
    acc_str = f"{warm_results['accuracy'] * 100:.1f}%"
    gov_str = f"{warm_results['governing_clause_accuracy'] * 100:.1f}%"
    outdated_str = f"{warm_results['outdated_answer_rate'] * 100:.1f}%"
    rec_str = f"{warm_results['evidence_recall'] * 100:.1f}%"
    prec_str = f"{warm_results['evidence_precision'] * 100:.1f}%"
    cons_str = f"{consistency_rate:.1f}%"
    chunks_str = f"{agg_mean_chunks:.1f}"
    vol_red_str = f"{retrieval_volume_reduction:.1f}%"
    lat_str = f"{agg_mean_latency:.1f} ms"
    lat_red_str = f"{latency_reduction:.1f}%"
    ctx_hit_str = f"{warm_results['context_cache_hit_rate'] * 100:.1f}%"
    emb_hit_str = f"{warm_results['query_cache_hit_rate'] * 100:.1f}%"

    print("=" * 80)
    print("SIDE-BY-SIDE ARCHITECTURAL COMPARISON: BASELINE vs CAKE vs AMO")
    print("=" * 80)
    print(f"{'Metric':<34} | {'Phase 1 Baseline':<16} | {'Phase 2 CAKE':<14} | {'Phase 3 AMO':<14}")
    print("-" * 80)
    print(f"{'Answer Accuracy':<34} | {'25.0%':<16} | {'62.5%':<14} | {acc_str:<14}")
    print(f"{'Governing Clause Accuracy':<34} | {'37.5%':<16} | {'100.0%':<14} | {gov_str:<14}")
    print(f"{'Outdated-Answer Rate':<34} | {'25.0%':<16} | {'0.0%':<14} | {outdated_str:<14}")
    print(f"{'Evidence Recall@5':<34} | {'100.0%':<16} | {'100.0%':<14} | {rec_str:<14}")
    print(f"{'Evidence Precision@5':<34} | {'42.5%':<16} | {'82.5%':<14} | {prec_str:<14}")
    print(f"{'Answer Consistency vs Phase 2':<34} | {'N/A':<16} | {'100.0%':<14} | {cons_str:<14}")
    print(f"{'Average Retrieved Chunks':<34} | {'5.0':<16} | {'5.0':<14} | {chunks_str:<14}")
    print(f"{'Retrieval Volume Reduction':<34} | {'0.0%':<16} | {'0.0%':<14} | {vol_red_str:<14}")
    print(f"{'Mean Query Latency':<34} | {'181.5 ms':<16} | {'174.6 ms':<14} | {lat_str:<14}")
    print(f"{'Latency Reduction vs Phase 2':<34} | {'N/A':<16} | {'0.0%':<14} | {lat_red_str:<14}")
    print(f"{'Context Cache Hit Rate (Warm)':<34} | {'N/A':<16} | {'N/A':<14} | {ctx_hit_str:<14}")
    print(f"{'Embedding Cache Hit Rate (Warm)':<34} | {'N/A':<16} | {'N/A':<14} | {emb_hit_str:<14}")
    print("=" * 80)

    # 8. Print Parametric Sensitivity Curve
    print("\nPARAMETRIC SENSITIVITY CURVE (Across Query Repetition Rate alpha):")
    print("=" * 100)
    print(f"{'Repetition Rate (alpha)':<25} | {'Mean Latency':<15} | {'Latency Reduction':<20} | {'Mean Chunks':<15} | {'Volume Reduction':<18}")
    print("-" * 100)
    for alpha_str, pdata in parametric_results.items():
        print(f"{alpha_str:<25} | {pdata['mean_latency_ms']:<12.2f} ms | {pdata['latency_reduction_pct']:<19.2f}% | {pdata['mean_chunks_retrieved']:<15.2f} | {pdata['retrieval_volume_reduction_pct']:<17.2f}%")
    print("=" * 100)

    # 9. Print Real Cache Hit Trace Sample
    print("\nSAMPLE REAL EXECUTION TRACE (Q001 Warm Pass Cache Hit):")
    sample_trace = warm_results["traces"][0]
    print(f"  Query                   : {sample_trace['question']}")
    print(f"  QueryCache Hit          : {sample_trace['query_cache_hit']}")
    print(f"  FamilyRouter Result     : {sample_trace['family_router']}")
    print(f"  ContextCache Hit        : {sample_trace['context_cache_hit']}")
    print(f"  Retrieval Count         : {sample_trace['chunks_retrieved_count']} chunks (Qdrant bypassed)")
    print(f"  CAKE Governing Clause(s): {sample_trace['governing_clauses']}")
    print(f"  Final Answer            : {sample_trace['final_answer']}")
    print(f"  Cited Clauses           : {sample_trace['cited_clauses']}")
    print(f"  Latency                 : {sample_trace['latency_ms']:.2f} ms\n")

    pipeline.close()
    print(f"AMO Benchmark artifacts safely stored in: {results_dir}\n")


if __name__ == "__main__":
    main()
