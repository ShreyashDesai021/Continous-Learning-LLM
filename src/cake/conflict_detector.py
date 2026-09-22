"""Conflict Detector identifying unresolved disagreements between active policy clauses."""

from typing import List, Optional
from pydantic import BaseModel, Field
from src.core.models import RetrievedChunk
from src.cake.temporal_selector import select_governing_clause


class ConflictResult(BaseModel):
    has_conflict: bool
    competing_clauses: List[RetrievedChunk] = Field(default_factory=list)
    governing_clause: Optional[RetrievedChunk] = None
    superseded_clauses: List[RetrievedChunk] = Field(default_factory=list)
    future_clauses: List[RetrievedChunk] = Field(default_factory=list)


def detect_conflicts(
    family_chunks: List[RetrievedChunk],
    snapshot_at: str,
    target_time: Optional[str] = None
) -> ConflictResult:
    """
    Detects whether active clauses for a family have an unresolved conflict.
    If a clear governing clause exists via temporal supersession or authority, has_conflict is False.
    If multiple active clauses disagree without an authoritative winner, returns has_conflict=True.
    """
    if not family_chunks:
        return ConflictResult(has_conflict=False)

    governing, superseded, future = select_governing_clause(
        family_chunks, 
        snapshot_at=snapshot_at, 
        target_time=target_time
    )

    if governing is not None:
        return ConflictResult(
            has_conflict=False,
            governing_clause=governing,
            superseded_clauses=superseded,
            future_clauses=future
        )

    # If governing could not be determined, check active clauses for genuine disagreement
    active = [c for c in family_chunks if c not in future]
    
    # If more than 1 active clause with conflicting values/text
    if len(active) >= 2:
        # Check if clauses say different numbers/values
        texts = [c.text.lower() for c in active]
        # If texts are distinct and not exact duplicates
        if len(set(texts)) > 1:
            return ConflictResult(
                has_conflict=True,
                competing_clauses=active,
                governing_clause=None,
                future_clauses=future
            )

    return ConflictResult(
        has_conflict=False,
        governing_clause=active[0] if active else None,
        future_clauses=future
    )
