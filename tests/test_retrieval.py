"""Tests for clause chunking and vector retrieval."""

import numpy as np
import pytest
from src.core.models import DocumentMetadata, DocumentScope, DocumentAuthority, DocumentClause
from src.baseline.chunker import parse_clauses_from_text, chunk_document
from src.baseline.vector_store import LocalQdrantStore


def test_parse_clauses():
    text = (
        "[C1] First policy clause text.\n"
        "[C2] Second policy clause text.\n"
        "[C3] Third policy clause text."
    )
    clauses = parse_clauses_from_text(text)
    assert len(clauses) == 3
    assert clauses[0]["clause_number"] == "C1"
    assert clauses[0]["text"] == "First policy clause text."
    assert clauses[2]["clause_number"] == "C3"


def test_chunk_document_metadata_enrichment():
    meta = DocumentMetadata(
        document_id="D101",
        policy_family="remote_work",
        version="1.0",
        publication_date="2024-12-15",
        ingested_at="2024-12-16T00:00:00Z",
        effective_from="2025-01-01",
        effective_to=None,
        department="HR",
        scope=DocumentScope(locations=["Pune", "Mumbai"], employee_category="permanent"),
        source_id="HR_01",
        authority=DocumentAuthority(issuer="HR Director", source_class="corporate_policy", approval_status="approved"),
        event_type="INITIAL_IMPORT",
        title="Remote Working Policy v1.0"
    )
    content = "[C1] Permanent employees may work remotely for a maximum of two days per week."
    doc_record = {"metadata": meta, "content": content}
    chunks = chunk_document(doc_record)

    assert len(chunks) == 1
    assert chunks[0].clause_id == "D101#C1"
    assert "HR Director (approved" in chunks[0].formatted_chunk
    assert "EFFECTIVE_FROM: 2025-01-01" in chunks[0].formatted_chunk


def test_vector_store_indexing_and_search():
    store = LocalQdrantStore(storage_path=None, dimension=4)  # in-memory store
    chunks = [
        DocumentClause(
            clause_id="D101#C1",
            document_id="D101",
            clause_number="C1",
            text="Two remote days",
            formatted_chunk="[DOC: D101] Two remote days",
            metadata={"doc": "D101"}
        ),
        DocumentClause(
            clause_id="D102#C1",
            document_id="D102",
            clause_number="C1",
            text="Three remote days",
            formatted_chunk="[DOC: D102] Three remote days",
            metadata={"doc": "D102"}
        )
    ]
    # Synthetic orthogonal normalized vectors
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)

    store.index_chunks("test_col", chunks, vecs)
    
    # Query aligned with first vector
    query_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = store.search("test_col", query_vec, top_k=1)

    assert len(results) == 1
    assert results[0].clause_id == "D101#C1"
    assert results[0].score > 0.8
