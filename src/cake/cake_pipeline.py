"""CAKE Pipeline implementing Conflict-Aware and Version-Aware Retrieval."""

import time
from typing import List, Dict, Any, Tuple
from src.core.models import Question, BaselineOutput, RetrievedChunk
from src.baseline.pipeline import ConventionalRAGPipeline
from src.cake.family_resolver import group_by_policy_family
from src.cake.conflict_detector import detect_conflicts, ConflictResult


class CAKEPipeline(ConventionalRAGPipeline):
    """
    RAG Pipeline with Conflict-Aware Knowledge Evolution (CAKE) layer.
    Filters retrieved chunks using deterministic temporal and authority resolution
    before passing evidence to the generator.
    """

    def resolve_cake_context(
        self,
        retrieved_chunks: List[RetrievedChunk],
        snapshot_at: str,
        question_text: str = ""
    ) -> Tuple[List[RetrievedChunk], List[str], List[str], List[str]]:
        """
        Applies CAKE resolution across retrieved chunks:
        1. Groups chunks by policy family and scope.
        2. Selects governing clauses for the snapshot/target timestamp.
        3. Suppresses superseded historical versions for current queries.
        4. Excludes future-effective versions from current governing context.
        5. Preserves competing clauses for genuine unresolved conflicts.
        6. Preserves complete chronology for history chain queries.
        """
        from src.cake.temporal_selector import detect_target_date

        family_groups = group_by_policy_family(retrieved_chunks)
        target_time = detect_target_date(question_text, snapshot_at)
        is_history_chain = any(kw in question_text.lower() for kw in ["sequence", "history", "timeline"])

        resolved_chunks: List[RetrievedChunk] = []
        governing_ids: List[str] = []
        suppressed_ids: List[str] = []
        future_ids: List[str] = []

        for family_key, chunks in family_groups.items():
            if is_history_chain:
                # Retain all active versions in chronological sequence
                for c in chunks:
                    resolved_chunks.append(c)
                    governing_ids.append(c.clause_id)
                continue

            conflict_res: ConflictResult = detect_conflicts(
                chunks, 
                snapshot_at=snapshot_at,
                target_time=target_time
            )

            # Record suppressed superseded clauses
            for sup in conflict_res.superseded_clauses:
                suppressed_ids.append(sup.clause_id)

            # Record future-effective clauses
            for fut in conflict_res.future_clauses:
                future_ids.append(fut.clause_id)

            if conflict_res.has_conflict:
                # Genuinely unresolved conflict: preserve competing clauses so model can report disagreement
                for comp in conflict_res.competing_clauses:
                    resolved_chunks.append(comp)
            else:
                if conflict_res.governing_clause is not None:
                    resolved_chunks.append(conflict_res.governing_clause)
                    governing_ids.append(conflict_res.governing_clause.clause_id)

        # Preserve relative ranking order from original retrieval
        final_chunks = [c for c in retrieved_chunks if c in resolved_chunks]
        return final_chunks, governing_ids, suppressed_ids, future_ids

    def run_query(self, question: Question) -> BaselineOutput:
        """Executes CAKE-filtered RAG workflow for a given question."""
        start_time = time.perf_counter()

        # 1. Snapshot index lookup
        col_name = self.ensure_snapshot_indexed(question.snapshot_at)

        # 2. Dense query embedding & Top-K retrieval
        q_vec = self.embedding_client.embed_query(question.question)
        raw_retrieved_chunks = self.vector_store.search(
            collection_name=col_name,
            query_vector=q_vec,
            top_k=self.top_k
        )

        # 3. CAKE Conflict & Temporal Resolution Layer
        (
            cake_chunks,
            governing_ids,
            suppressed_ids,
            future_ids
        ) = self.resolve_cake_context(
            raw_retrieved_chunks, 
            question.snapshot_at,
            question_text=question.question
        )

        # 4. LLM Generation conditioned strictly on CAKE-resolved context
        gen_result = self.generator.generate(
            question=question.question,
            chunks=cake_chunks,
            snapshot_at=question.snapshot_at,
            asked_at=question.asked_at,
            user_context=question.user_context.model_dump()
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        output = BaselineOutput(
            question_id=question.question_id,
            question=question.question,
            snapshot_at=question.snapshot_at,
            asked_at=question.asked_at,
            answer=gen_result.get("answer", ""),
            cited_clauses=gen_result.get("cited_clauses", []),
            retrieved_chunks=cake_chunks,
            model_name=self.generator.model_name,
            embedding_model=self.embedding_client.model_name,
            top_k=self.top_k,
            latency_ms=latency_ms
        )
        # Store CAKE metadata in extra attributes for evaluation
        output.__dict__["raw_retrieved_chunks"] = raw_retrieved_chunks
        output.__dict__["governing_ids"] = governing_ids
        output.__dict__["suppressed_ids"] = suppressed_ids
        output.__dict__["future_ids"] = future_ids

        return output
