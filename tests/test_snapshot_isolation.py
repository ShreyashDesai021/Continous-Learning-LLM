"""Tests for temporal snapshot isolation."""

from pathlib import Path
import pytest
from src.core.snapshot import get_eligible_documents, parse_iso_timestamp

DATA_DIR = Path("d:/BTech Project Antigravity/data")
META_PATH = DATA_DIR / "inference" / "development" / "metadata.jsonl"
DOCS_DIR = DATA_DIR / "inference" / "development" / "documents"


def test_early_snapshot_excludes_future_documents():
    """At 2026-02-10, documents ingested later (e.g. D103 in June, D503 in March) must not be eligible."""
    snapshot_at = "2026-02-10T23:59:59Z"
    eligible = get_eligible_documents(snapshot_at, META_PATH, DOCS_DIR)
    eligible_ids = set(doc["metadata"].document_id for doc in eligible)

    # D101, D102, D201, D202, D203, D401, D501, D502 were ingested on or before Feb 10
    assert "D101" in eligible_ids
    assert "D102" in eligible_ids
    assert "D201" in eligible_ids
    assert "D501" in eligible_ids

    # D103 was ingested 2026-06-16 -> MUST NOT BE INCLUDED
    assert "D103" not in eligible_ids
    # D503 was ingested 2026-03-21 -> MUST NOT BE INCLUDED
    assert "D503" not in eligible_ids
    # D403 was ingested 2026-04-16 -> MUST NOT BE INCLUDED
    assert "D403" not in eligible_ids


def test_june_snapshot_includes_up_to_june_documents():
    """At 2026-06-20, documents ingested up to June 20 must be included, but not July/August."""
    snapshot_at = "2026-06-20T23:59:59Z"
    eligible = get_eligible_documents(snapshot_at, META_PATH, DOCS_DIR)
    eligible_ids = set(doc["metadata"].document_id for doc in eligible)

    assert "D103" in eligible_ids
    assert "D104" in eligible_ids
    assert "D204" in eligible_ids
    assert len(eligible_ids) == 20  # All 20 development documents were ingested by June 19


def test_timestamp_parsing():
    """Verify ISO timestamps parse with UTC timezone awareness."""
    dt1 = parse_iso_timestamp("2026-02-10T23:59:59Z")
    dt2 = parse_iso_timestamp("2026-02-11T00:00:00Z")
    assert dt1 < dt2
    assert dt1.tzinfo is not None
