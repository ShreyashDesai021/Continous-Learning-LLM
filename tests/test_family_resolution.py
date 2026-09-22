"""Tests for deterministic policy family and sub-scope grouping in CAKE."""

import pytest
from src.core.models import RetrievedChunk
from src.cake.family_resolver import group_by_policy_family, get_rule_key


def test_remote_work_grouping():
    chunks = [
        RetrievedChunk(clause_id="D101#C1", document_id="D101", score=0.65, text="2 days per week", metadata={"policy_family": "remote_work"}),
        RetrievedChunk(clause_id="D102#C1", document_id="D102", score=0.68, text="3 days per week", metadata={"policy_family": "remote_work"}),
        RetrievedChunk(clause_id="D103#C1", document_id="D103", score=0.70, text="4 days per week", metadata={"policy_family": "remote_work"}),
    ]
    groups = group_by_policy_family(chunks)
    assert len(groups) == 1
    assert "remote_work_max_days" in groups
    assert len(groups["remote_work_max_days"]) == 3


def test_office_meals_scope_separation():
    """Pune and Mumbai meal ceilings must NOT be conflated into the same family group."""
    chunks = [
        RetrievedChunk(clause_id="D201#C1", document_id="D201", score=0.8, text="Pune meal INR 250", metadata={"policy_family": "office_meals", "locations": ["Pune"]}),
        RetrievedChunk(clause_id="D202#C1", document_id="D202", score=0.82, text="Mumbai meal INR 350", metadata={"policy_family": "office_meals", "locations": ["Mumbai"]}),
        RetrievedChunk(clause_id="D204#C1", document_id="D204", score=0.79, text="Mumbai meal INR 400", metadata={"policy_family": "office_meals", "locations": ["Mumbai"]}),
    ]
    groups = group_by_policy_family(chunks)
    assert "office_meals_pune" in groups
    assert "office_meals_mumbai" in groups
    assert len(groups["office_meals_pune"]) == 1
    assert len(groups["office_meals_mumbai"]) == 2


def test_deployment_scope_separation():
    """Development deployment and production deployment must remain separate."""
    chunks = [
        RetrievedChunk(clause_id="D501#C1", document_id="D501", score=0.8, text="Production requires 2 reviewers", metadata={"policy_family": "deployment", "conditions": ["production deployments"]}),
        RetrievedChunk(clause_id="D502#C1", document_id="D502", score=0.85, text="Development requires 1 reviewer", metadata={"policy_family": "deployment", "conditions": ["development-environment deployments"]}),
    ]
    groups = group_by_policy_family(chunks)
    assert "deployment_reviewers_prod" in groups
    assert "deployment_reviewers_dev" in groups
    assert groups["deployment_reviewers_prod"][0].clause_id == "D501#C1"
    assert groups["deployment_reviewers_dev"][0].clause_id == "D502#C1"
