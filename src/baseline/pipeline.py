"""End-to-end Conventional RAG baseline pipeline."""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.models import Question, BaselineOutput
from src.core.snapshot import get_eligible_documents
from src.baseline.chunker import chunk_all_eligible_documents
from src.baseline.embeddings import BGEM3EmbeddingClient
from src.baseline.vector_store import LocalQdrantStore
from src.baseline.generator import LocalQwenGenerator


class ConventionalRAGPipeline:
    """Conventional RAG pipeline with snapshot isolation, BGE-M3, Qdrant, and Qwen."""

    def __init__(
        self,
        data_dir: str = "data",
        storage_dir: str = "runtime/research",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        top_k: int = 5,
        mock_generator: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.storage_dir = Path(storage_dir)
        self.top_k = top_k
        self.mock_generator = mock_generator

        self.metadata_path = self.data_dir / "inference" / "development" / "metadata.jsonl"
        self.docs_dir = self.data_dir / "inference" / "development" / "documents"

        # Initialize components
        self.embedding_client = BGEM3EmbeddingClient(
            model_name=embedding_model,
            cache_dir=str(self.storage_dir / "models")
        )
        self.vector_store = LocalQdrantStore(
            storage_path=str(self.storage_dir / "qdrant_storage")
        )
        self.generator = LocalQwenGenerator(
            model_name=llm_model,
            cache_dir=str(self.storage_dir / "models"),
            max_new_tokens=128,
            mock_mode=mock_generator
        )

        self._indexed_snapshots = set()

    def preload_models(self) -> None:
        """Preloads embedding model and LLM into memory so queries run at full speed."""
        _ = self.embedding_client.model
        if not self.mock_generator:
            self.generator._load_model()

    def _get_snapshot_collection_name(self, snapshot_at: str) -> str:
        """Creates unique Qdrant collection name for each repository snapshot."""
        h = hashlib.md5(snapshot_at.encode("utf-8")).hexdigest()[:10]
        return f"snap_{h}"

    def ensure_snapshot_indexed(self, snapshot_at: str) -> str:
        """Indexes all eligible documents for a snapshot if not already indexed."""
        col_name = self._get_snapshot_collection_name(snapshot_at)
        if col_name in self._indexed_snapshots:
            return col_name

        eligible_docs = get_eligible_documents(
            snapshot_at=snapshot_at,
            metadata_path=self.metadata_path,
            documents_dir=self.docs_dir
        )
        chunks = chunk_all_eligible_documents(eligible_docs)

        chunk_texts = [c.formatted_chunk for c in chunks]
        embeddings = self.embedding_client.embed_texts(chunk_texts)

        self.vector_store.index_chunks(
            collection_name=col_name,
            chunks=chunks,
            embeddings=embeddings
        )
        self._indexed_snapshots.add(col_name)
        return col_name

    def run_query(self, question: Question) -> BaselineOutput:
        """Executes full RAG workflow for a given question under snapshot constraints."""
        start_time = time.perf_counter()

        # 1. Snapshot index lookup
        col_name = self.ensure_snapshot_indexed(question.snapshot_at)

        # 2. Query embedding & Top-K retrieval
        q_vec = self.embedding_client.embed_query(question.question)
        retrieved_chunks = self.vector_store.search(
            collection_name=col_name,
            query_vector=q_vec,
            top_k=self.top_k
        )

        # 3. LLM generation
        gen_result = self.generator.generate(
            question=question.question,
            chunks=retrieved_chunks,
            snapshot_at=question.snapshot_at,
            asked_at=question.asked_at,
            user_context=question.user_context.model_dump()
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return BaselineOutput(
            question_id=question.question_id,
            question=question.question,
            snapshot_at=question.snapshot_at,
            asked_at=question.asked_at,
            answer=gen_result.get("answer", ""),
            cited_clauses=gen_result.get("cited_clauses", []),
            retrieved_chunks=retrieved_chunks,
            model_name=self.generator.model_name,
            embedding_model=self.embedding_client.model_name,
            top_k=self.top_k,
            latency_ms=latency_ms
        )

    def close(self) -> None:
        """Closes resources cleanly."""
        if hasattr(self, "vector_store") and self.vector_store is not None:
            self.vector_store.close()

