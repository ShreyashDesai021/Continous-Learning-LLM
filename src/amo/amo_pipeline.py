"""Adaptive Memory Optimization (AMO) Pipeline for CLLM."""

import time
from typing import List, Dict, Any, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.core.models import Question, BaselineOutput, RetrievedChunk
from src.cake.cake_pipeline import CAKEPipeline
from src.amo.family_router import FamilyRouter, RoutingResult
from src.amo.query_cache import QueryCache
from src.amo.context_cache import ContextCache, CachedCAKEContext, compute_intent_signature


class AMOPipeline(CAKEPipeline):
    """
    RAG Pipeline with Adaptive Memory Optimization (AMO) layer sitting between retrieval and CAKE.
    Optimizations:
    1. QueryCache: Sub-millisecond BGE-M3 query vector reuse via persistent MD5 LRU caching.
    2. FamilyRouter: Deterministic keyword routing with confidence scoring and safe fallback.
    3. Payload-Filtered Retrieval: Shortlists Qdrant search by metadata.policy_family.
    4. ContextCache (Query Pattern Memory): Caches resolved CAKE contexts by (snapshot, family, intent_signature).
    """

    def __init__(
        self,
        data_dir: str = "data",
        storage_dir: str = "runtime/research",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        top_k: int = 5,
        mock_generator: bool = False,
        router_confidence_threshold: float = 0.70
    ):
        super().__init__(
            data_dir=data_dir,
            storage_dir=storage_dir,
            embedding_model=embedding_model,
            llm_model=llm_model,
            top_k=top_k,
            mock_generator=mock_generator
        )

        self.family_router = FamilyRouter(confidence_threshold=router_confidence_threshold)
        self.query_cache = QueryCache(
            cache_dir=str(self.storage_dir / "query_embedding_cache"),
            maxsize=1000
        )
        self.context_cache = ContextCache(
            cache_dir=str(self.storage_dir / "context_cache"),
            max_entries_per_family=200
        )

    def run_query(self, question: Question) -> BaselineOutput:
        """Executes AMO-optimized RAG workflow for a given question."""
        start_time = time.perf_counter()

        # 1. Query Embedding Cache Check
        cached_q_vec = self.query_cache.get(question.question)
        if cached_q_vec is not None:
            q_vec = cached_q_vec
            query_cache_hit = True
        else:
            q_vec = self.embedding_client.embed_query(question.question)
            self.query_cache.set(question.question, q_vec)
            query_cache_hit = False

        # 2. Deterministic Family Routing
        user_ctx_dict = question.user_context.model_dump() if question.user_context else {}
        routing_res = self.family_router.route_query(question.question, user_ctx_dict)
        candidate_family = routing_res.candidate_family

        # 3. Context Cache Check (Query Pattern Memory) - BEFORE any snapshot indexing
        cached_ctx: Optional[CachedCAKEContext] = None
        if candidate_family:
            cached_ctx = self.context_cache.get(
                snapshot_at=question.snapshot_at,
                policy_family=candidate_family,
                question_text=question.question,
                user_context=user_ctx_dict
            )

        if cached_ctx is not None:
            # CONTEXT CACHE HIT: 100% bypass of Qdrant dense vector search & CAKE resolution & snapshot indexing
            context_cache_hit = True
            raw_retrieved_chunks = []
            cake_chunks = cached_ctx.governing_chunks
            governing_ids = cached_ctx.governing_ids
            suppressed_ids = cached_ctx.suppressed_ids
            future_ids = cached_ctx.future_ids
            has_conflict = cached_ctx.has_conflict
            retrieval_method = "context_cache_hit"
        else:
            # CONTEXT CACHE MISS: Index snapshot only now when retrieval is truly needed
            context_cache_hit = False
            col_name = self.ensure_snapshot_indexed(question.snapshot_at)

            if candidate_family is not None and not routing_res.fallback_to_global:
                # Shortlist via Qdrant native payload filter
                family_filter = Filter(
                    must=[FieldCondition(key="metadata.policy_family", match=MatchValue(value=candidate_family))]
                )
                raw_retrieved_chunks = self.vector_store.search_filtered(
                    collection_name=col_name,
                    query_vector=q_vec,
                    top_k=self.top_k,
                    query_filter=family_filter
                )
                retrieval_method = f"family_filtered_{candidate_family}"

                # Safety fallback: if filtered search yields empty or ultra-low score, fallback to global
                if not raw_retrieved_chunks or raw_retrieved_chunks[0].score < 0.35:
                    raw_retrieved_chunks = self.vector_store.search(
                        collection_name=col_name,
                        query_vector=q_vec,
                        top_k=self.top_k
                    )
                    retrieval_method = "fallback_global_low_confidence"
            else:
                # Global search fallback
                raw_retrieved_chunks = self.vector_store.search(
                    collection_name=col_name,
                    query_vector=q_vec,
                    top_k=self.top_k
                )
                retrieval_method = "global_unfiltered"

            # Check for Out-of-Domain (OOD) query: Abstain if unrecognized family and score < 0.52, or score < 0.40
            max_score = max((c.score for c in raw_retrieved_chunks), default=0.0)
            is_ood = (candidate_family is None and max_score < 0.52) or (max_score < 0.40)
            if is_ood:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                output = BaselineOutput(
                    question_id=question.question_id,
                    question=question.question,
                    snapshot_at=question.snapshot_at,
                    asked_at=question.asked_at,
                    answer="I could not find any relevant enterprise policy governing the requested query in the current knowledge base.",
                    cited_clauses=[],
                    retrieved_chunks=[],
                    model_name=self.generator.model_name,
                    embedding_model=self.embedding_client.model_name,
                    top_k=self.top_k,
                    latency_ms=latency_ms
                )
                output.__dict__["raw_retrieved_chunks"] = raw_retrieved_chunks
                output.__dict__["governing_ids"] = []
                output.__dict__["suppressed_ids"] = []
                output.__dict__["future_ids"] = []
                output.__dict__["has_conflict"] = False
                output.__dict__["amo_query_cache_hit"] = query_cache_hit
                output.__dict__["amo_context_cache_hit"] = False
                output.__dict__["amo_retrieval_method"] = "out_of_domain_abstention"
                output.__dict__["amo_chunks_retrieved_count"] = len(raw_retrieved_chunks)
                output.__dict__["amo_routing_result"] = routing_res
                return output

            # 4. CAKE Conflict & Temporal Resolution Layer
            (
                cake_chunks,
                governing_ids,
                suppressed_ids,
                future_ids,
                has_conflict
            ) = self.resolve_cake_context(
                raw_retrieved_chunks,
                question.snapshot_at,
                question_text=question.question
            )

            # Store resolved context in ContextCache for future reuse
            if candidate_family:
                intent_sig = compute_intent_signature(
                    question.question,
                    question.snapshot_at,
                    user_ctx_dict
                )
                new_cached_ctx = CachedCAKEContext(
                    governing_chunks=cake_chunks,
                    governing_ids=governing_ids,
                    suppressed_ids=suppressed_ids,
                    future_ids=future_ids,
                    has_conflict=has_conflict,
                    policy_family=candidate_family,
                    snapshot_at=question.snapshot_at,
                    intent_signature=intent_sig
                )
                self.context_cache.set(
                    snapshot_at=question.snapshot_at,
                    policy_family=candidate_family,
                    question_text=question.question,
                    context=new_cached_ctx,
                    user_context=user_ctx_dict
                )

        # 5. LLM Generation conditioned strictly on CAKE-resolved context
        gen_result = self.generator.generate(
            question=question.question,
            chunks=cake_chunks,
            snapshot_at=question.snapshot_at,
            asked_at=question.asked_at,
            user_context=user_ctx_dict
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

        # AMO and CAKE Telemetry
        output.__dict__["raw_retrieved_chunks"] = raw_retrieved_chunks
        output.__dict__["governing_ids"] = governing_ids
        output.__dict__["suppressed_ids"] = suppressed_ids
        output.__dict__["future_ids"] = future_ids
        output.__dict__["has_conflict"] = has_conflict
        output.__dict__["amo_query_cache_hit"] = query_cache_hit
        output.__dict__["amo_context_cache_hit"] = context_cache_hit
        output.__dict__["amo_retrieval_method"] = retrieval_method
        output.__dict__["amo_chunks_retrieved_count"] = len(raw_retrieved_chunks)
        output.__dict__["amo_routing_result"] = routing_res

        return output
