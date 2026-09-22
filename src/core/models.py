"""Data models for Enterprise CLLM Phase 1 Baseline."""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class DocumentScope(BaseModel):
    locations: List[str] = Field(default_factory=list)
    employee_category: str = "all"
    conditions: List[str] = Field(default_factory=list)


class DocumentAuthority(BaseModel):
    issuer: str
    source_class: str
    approval_status: str  # approved, unapproved, unknown


class DocumentMetadata(BaseModel):
    document_id: str
    policy_family: str
    version: Optional[str] = None
    publication_date: Optional[str] = None
    ingested_at: str
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    department: str
    scope: DocumentScope
    source_id: str
    authority: DocumentAuthority
    event_type: str  # INITIAL_IMPORT, ADDED, REUPLOADED
    title: str


class DocumentClause(BaseModel):
    clause_id: str          # e.g., "D101#C1"
    document_id: str        # e.g., "D101"
    clause_number: str      # e.g., "C1"
    text: str               # Raw clause text
    formatted_chunk: str    # Metadata-enriched chunk text for embeddings
    metadata: Dict[str, Any]


class UserContext(BaseModel):
    locations: List[str] = Field(default_factory=list)
    employee_category: str = "all"
    conditions: List[str] = Field(default_factory=list)


class Question(BaseModel):
    question_id: str
    policy_family: str
    question: str
    snapshot_at: str
    asked_at: str
    user_context: UserContext


class GoldAnswer(BaseModel):
    question_id: str
    policy_family: str
    gold_answer: str
    required_claims: List[str]
    forbidden_claims: List[str]
    acceptable_evidence_sets: List[List[str]]
    question_category: str
    expected_difficulty: str


class RetrievedChunk(BaseModel):
    clause_id: str
    document_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class BaselineOutput(BaseModel):
    question_id: str
    question: str
    snapshot_at: str
    asked_at: str
    answer: str
    cited_clauses: List[str]
    retrieved_chunks: List[RetrievedChunk]
    model_name: str
    embedding_model: str
    top_k: int
    latency_ms: float


class QuestionEvaluation(BaseModel):
    question_id: str
    policy_family: str
    question_category: str
    answer_correctness: bool
    required_claim_coverage: float
    forbidden_claim_violations: int
    evidence_recall: float
    evidence_precision: float
    outdated_answer_detected: bool
    conflict_handled_correctly: bool
    latency_ms: float
    generated_answer: str
    gold_answer: str
    retrieved_clauses: List[str]


class EvaluationSummary(BaseModel):
    total_questions: int
    mean_accuracy: float
    mean_required_claim_coverage: float
    forbidden_claim_violation_rate: float
    mean_evidence_recall: float
    mean_evidence_precision: float
    outdated_answer_rate: float
    conflict_handling_rate: float
    mean_latency_ms: float
    category_breakdown: Dict[str, Dict[str, Any]]
