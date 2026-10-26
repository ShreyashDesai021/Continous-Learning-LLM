"""Unit tests for FamilyRouter: lexicon scoring, confidence computation, and fallback."""

import pytest
from src.amo.family_router import FamilyRouter, RoutingResult


def test_routing_all_five_policy_families():
    router = FamilyRouter(confidence_threshold=0.70)

    # 1. remote_work
    res_rw = router.route_query("How many remote days per week are allowed today?")
    assert res_rw.candidate_family == "remote_work"
    assert res_rw.confidence >= 0.70
    assert not res_rw.fallback_to_global

    # 2. office_meals
    res_om = router.route_query("What is the meal ceiling for Pune employees working late?")
    assert res_om.candidate_family == "office_meals"
    assert res_om.confidence >= 0.70
    assert not res_om.fallback_to_global

    # 3. support_callback
    res_sc = router.route_query("What callback deadline applies to standard-plan incidents?")
    assert res_sc.candidate_family == "support_callback"
    assert res_sc.confidence >= 0.70
    assert not res_sc.fallback_to_global

    # 4. backup_retention
    res_br = router.route_query("What is the production-backup retention period?")
    assert res_br.candidate_family == "backup_retention"
    assert res_br.confidence >= 0.70
    assert not res_br.fallback_to_global

    # 5. deployment
    res_dp = router.route_query("How many reviewers are required for a production deployment?")
    assert res_dp.candidate_family == "deployment"
    assert res_dp.confidence >= 0.70
    assert not res_dp.fallback_to_global


def test_routing_fallback_on_unrecognized_query():
    router = FamilyRouter(confidence_threshold=0.70)
    res = router.route_query("What is the dress code in the cafeteria on Fridays?")
    assert res.candidate_family is None
    assert res.confidence == 0.0
    assert res.fallback_to_global
    assert res.reason == "no_keyword_match"


def test_routing_fallback_on_ambiguity_tie():
    router = FamilyRouter(confidence_threshold=0.70)
    # Query with equal weight keywords from two families
    res = router.route_query("Tell me about backup and deployment procedures.")
    assert res.fallback_to_global
    assert res.confidence <= 0.70


def test_routing_multi_match_with_clear_dominant_winner():
    router = FamilyRouter(confidence_threshold=0.70)
    # Multiple remote-work terms vs one incidental term
    res = router.route_query("What is the remote-work allowance for telecommute and wfh staff with backup laptop?")
    assert res.candidate_family == "remote_work"
    assert res.confidence >= 0.70
    assert not res.fallback_to_global


def test_routing_fallback_on_diluted_multi_family_match():
    """When top score is 3.0, but 3 other families each have 1.0 (total 6.0), conf=0.50 < 0.70 -> fallback."""
    router = FamilyRouter(confidence_threshold=0.70)
    # remote-work terms (wfh, telecommute, hybrid = 3.0) + meals (lunch = 1.0) + backup (archive = 1.0) + deploy (release = 1.0)
    query = "For wfh, telecommute, and hybrid staff, what are the lunch rules during an archive release?"
    res = router.route_query(query)
    assert res.fallback_to_global, f"Expected fallback due to multi-family dilution, got candidate: {res.candidate_family}"
    assert res.confidence < 0.70
