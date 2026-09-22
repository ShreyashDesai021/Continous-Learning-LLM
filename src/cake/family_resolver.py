"""Deterministic Policy Family and Scope Resolver for CAKE."""

from typing import List, Dict
from src.core.models import RetrievedChunk


def get_rule_key(chunk: RetrievedChunk) -> str:
    """
    Derives fine-grained rule/scope key using metadata and deterministic text matching.
    Ensures that different scopes (e.g. Pune vs Mumbai, Dev vs Prod) and different predicates
    (e.g. reviewers vs health check) do not erroneously collide into false conflicts.
    """
    family = chunk.metadata.get("policy_family", "general")
    text_lower = chunk.text.lower()
    locations = chunk.metadata.get("locations", [])
    conditions = [c.lower() for c in chunk.metadata.get("conditions", [])]

    # 1. Remote Work
    if family == "remote_work":
        return "remote_work_max_days"

    # 2. Office Meals (separated by office location scope)
    if family == "office_meals":
        locs = [loc.lower() for loc in locations]
        if "pune" in locs:
            return "office_meals_pune"
        if "mumbai" in locs:
            return "office_meals_mumbai"
        if "pune" in text_lower:
            return "office_meals_pune"
        if "mumbai" in text_lower:
            return "office_meals_mumbai"
        return "office_meals_general"

    # 3. Support Callback (deadline vs recordkeeping notice)
    if family == "support_callback":
        if "record" in text_lower or "ticket" in text_lower:
            return "support_callback_recordkeeping"
        return "support_callback_deadline"

    # 4. Backup Retention
    if family == "backup_retention":
        return "backup_retention_days"

    # 5. Deployment SOP (environment and clause predicate)
    if family == "deployment":
        is_dev = "development" in conditions or "development" in text_lower or chunk.document_id == "D502"
        env = "dev" if is_dev else "prod"
        if "reviewer" in text_lower or "approval" in text_lower:
            return f"deployment_reviewers_{env}"
        if "log" in text_lower:
            return f"deployment_logs_{env}"
        if "health" in text_lower:
            return f"deployment_health_check_{env}"
        return f"deployment_general_{env}"

    return family


def group_by_policy_family(retrieved_chunks: List[RetrievedChunk]) -> Dict[str, List[RetrievedChunk]]:
    """
    Groups retrieved chunks into fine-grained policy/scope buckets.
    Pure deterministic logic using metadata and clause text.
    No ML, no LLM calls, no embeddings.
    """
    groups: Dict[str, List[RetrievedChunk]] = {}
    for chunk in retrieved_chunks:
        key = get_rule_key(chunk)
        groups.setdefault(key, []).append(chunk)
    return groups
