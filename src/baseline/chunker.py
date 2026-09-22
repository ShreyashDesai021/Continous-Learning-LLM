"""Clause-preserving document chunker with metadata enrichment for fair Conventional RAG."""

import re
from typing import List, Dict, Any
from src.core.models import DocumentMetadata, DocumentClause


def parse_clauses_from_text(text: str) -> List[Dict[str, str]]:
    """
    Parses clauses marked by [C1], [C2], etc. from document text.
    Returns list of dicts: {"clause_number": "C1", "text": "..."}
    """
    # Regex matching [C1], [C2], etc.
    pattern = r"\[(C\d+)\]\s*(.*?)(?=\[(?:C\d+)\]|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    clauses = []
    if matches:
        for c_num, c_text in matches:
            clean_text = c_text.strip()
            if clean_text:
                clauses.append({
                    "clause_number": c_num,
                    "text": clean_text
                })
    else:
        # Fallback if no [C#] tags found
        clauses.append({
            "clause_number": "C1",
            "text": text.strip()
        })
    return clauses


def chunk_document(doc_record: Dict[str, Any]) -> List[DocumentClause]:
    """
    Chunks a document into clause-level units while preserving metadata context.
    The formatted_chunk string includes the document header and metadata so the 
    dense embedding model and generator have full visibility into dates, scope, and authority.
    """
    meta: DocumentMetadata = doc_record["metadata"]
    content: str = doc_record["content"]
    
    raw_clauses = parse_clauses_from_text(content)
    chunks = []

    for item in raw_clauses:
        c_num = item["clause_number"]
        c_text = item["text"]
        clause_id = f"{meta.document_id}#{c_num}"

        # Construct enriched representation for embedding & retrieval
        formatted = (
            f"[DOC: {meta.document_id} | TITLE: {meta.title} | DEPT: {meta.department}]\n"
            f"[VERSION: {meta.version or 'N/A'} | PUBLISHED: {meta.publication_date or 'N/A'} | "
            f"EFFECTIVE_FROM: {meta.effective_from or 'N/A'} | EFFECTIVE_TO: {meta.effective_to or 'N/A'}]\n"
            f"[AUTHORITY: {meta.authority.issuer} ({meta.authority.approval_status}, {meta.authority.source_class})]\n"
            f"[SCOPE: locations={meta.scope.locations}, category={meta.scope.employee_category}, conditions={meta.scope.conditions}]\n"
            f"[CLAUSE: {clause_id}] {c_text}"
        )

        chunk = DocumentClause(
            clause_id=clause_id,
            document_id=meta.document_id,
            clause_number=c_num,
            text=c_text,
            formatted_chunk=formatted,
            metadata={
                "document_id": meta.document_id,
                "policy_family": meta.policy_family,
                "version": meta.version,
                "publication_date": meta.publication_date,
                "effective_from": meta.effective_from,
                "effective_to": meta.effective_to,
                "department": meta.department,
                "issuer": meta.authority.issuer,
                "approval_status": meta.authority.approval_status,
                "source_class": meta.authority.source_class,
                "locations": meta.scope.locations,
                "employee_category": meta.scope.employee_category,
                "conditions": meta.scope.conditions
            }
        )
        chunks.append(chunk)

    return chunks


def chunk_all_eligible_documents(eligible_docs: List[Dict[str, Any]]) -> List[DocumentClause]:
    """Chunks all eligible documents for a snapshot."""
    all_chunks = []
    for doc in eligible_docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
