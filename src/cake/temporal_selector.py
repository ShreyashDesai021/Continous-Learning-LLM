"""Temporal Rule Selector for CAKE selecting the governing clause for a snapshot."""

from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from src.core.models import RetrievedChunk


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parses date string or ISO timestamp into date object."""
    if not date_str:
        return None
    # If full ISO timestamp with T
    clean = date_str.split("T")[0]
    try:
        return datetime.strptime(clean, "%Y-%m-%d").date()
    except Exception:
        return None


def detect_target_date(question_text: str, default_snapshot: str) -> str:
    """
    Detects if question asks about a specific past year (e.g. 2025) or date.
    Returns effective date threshold for historical query evaluation.
    """
    import re
    m = re.search(r"\b(202[0-5])\b", question_text)
    if m:
        year = m.group(1)
        return f"{year}-12-31T23:59:59Z"
    return default_snapshot


def select_governing_clause(
    family_chunks: List[RetrievedChunk],
    snapshot_at: str,
    target_time: Optional[str] = None
) -> Tuple[Optional[RetrievedChunk], List[RetrievedChunk], List[RetrievedChunk]]:
    """
    Selects the governing clause for a policy family at snapshot_at (or target_time).
    
    Returns:
    (governing_chunk, superseded_chunks, future_chunks)
    - governing_chunk: The single clause in force with the most recent effective date
    - superseded_chunks: Active clauses displaced by a more recent approved revision
    - future_chunks: Clauses whose effective date is in the future relative to snapshot/target
    """
    if not family_chunks:
        return None, [], []

    eval_time = target_time if target_time else snapshot_at
    snapshot_date = parse_date(eval_time)
    if not snapshot_date:
        raise ValueError(f"Invalid snapshot timestamp: {snapshot_at}")

    # 1. Filter out unapproved suggestions/proposals if approved clauses exist
    approved_chunks = [
        c for c in family_chunks 
        if c.metadata.get("approval_status") == "approved"
    ]
    candidate_pool = approved_chunks if approved_chunks else family_chunks

    active_chunks: List[RetrievedChunk] = []
    future_chunks: List[RetrievedChunk] = []

    # 2. Separate active vs future-effective
    for chunk in candidate_pool:
        eff_from = parse_date(chunk.metadata.get("effective_from"))
        eff_to = parse_date(chunk.metadata.get("effective_to"))

        if eff_from is not None:
            if eff_from > snapshot_date:
                future_chunks.append(chunk)
                continue
            if eff_to is not None and eff_to <= snapshot_date:
                # Expired
                continue
        active_chunks.append(chunk)

    if not active_chunks:
        return None, [], future_chunks

    # 3. Handle cases where effective dates are missing (e.g. support callback wiki vs note)
    chunks_with_date = [c for c in active_chunks if parse_date(c.metadata.get("effective_from")) is not None]
    if not chunks_with_date:
        # If no dates available, cannot resolve temporal supersession
        # Check if they are duplicates or genuine conflicts
        return None, [], future_chunks

    # 4. Sort active dated clauses by effective_from descending
    chunks_with_date.sort(
        key=lambda c: parse_date(c.metadata.get("effective_from")),
        reverse=True
    )

    most_recent_date = parse_date(chunks_with_date[0].metadata.get("effective_from"))
    
    # All clauses sharing the most recent effective date
    top_candidates = [
        c for c in chunks_with_date 
        if parse_date(c.metadata.get("effective_from")) == most_recent_date
    ]

    # Clauses with older effective dates are superseded
    superseded = [
        c for c in chunks_with_date 
        if parse_date(c.metadata.get("effective_from")) < most_recent_date
    ]

    # If single top candidate (or duplicates carrying identical value)
    governing = top_candidates[0]
    # Any other top candidate that is an exact restatement can be considered equivalent
    for other in top_candidates[1:]:
        superseded.append(other)

    return governing, superseded, future_chunks
