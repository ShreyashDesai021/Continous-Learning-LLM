"""Snapshot filtering and document eligibility resolution."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.core.models import DocumentMetadata


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse ISO 8601 UTC timestamp string to aware datetime."""
    clean = ts_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(clean)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_all_metadata(metadata_path: Path) -> Dict[str, DocumentMetadata]:
    """Load metadata.jsonl into a dictionary keyed by document_id."""
    meta_dict = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            meta_dict[item["document_id"]] = DocumentMetadata(**item)
    return meta_dict


def get_eligible_documents(
    snapshot_at: str,
    metadata_path: Path,
    documents_dir: Path
) -> List[Dict[str, Any]]:
    """
    Returns all documents whose ingested_at <= snapshot_at.
    Guarantees no future document leakage into the snapshot.
    Each returned dictionary contains:
    - metadata: DocumentMetadata
    - content: raw document text
    """
    snapshot_dt = parse_iso_timestamp(snapshot_at)
    all_meta = load_all_metadata(metadata_path)

    eligible = []
    for doc_id, meta in all_meta.items():
        ingested_dt = parse_iso_timestamp(meta.ingested_at)
        if ingested_dt <= snapshot_dt:
            doc_file = documents_dir / f"{doc_id}.txt"
            if doc_file.exists():
                text = doc_file.read_text(encoding="utf-8")
                eligible.append({
                    "metadata": meta,
                    "content": text
                })
            else:
                raise FileNotFoundError(f"Document file missing: {doc_file}")

    return eligible
