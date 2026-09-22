"""Deterministic Policy Family Router for Adaptive Memory Optimization (AMO)."""

import re
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field


@dataclass
class RoutingResult:
    """Outcome of policy family routing."""
    candidate_family: Optional[str]
    confidence: float
    reason: str
    scores: Dict[str, float] = field(default_factory=dict)
    fallback_to_global: bool = False


# Domain lexicons for enterprise policy families
FAMILY_LEXICONS: Dict[str, List[str]] = {
    "remote_work": [
        "remote", "wfh", "work from home", "telecommute", "remote-work",
        "remote working", "hybrid", "remote days"
    ],
    "office_meals": [
        "meal", "meals", "dinner", "food", "lunch", "late-working",
        "meal allowance", "reimbursement", "meal ceiling", "evening meal", "21:00"
    ],
    "support_callback": [
        "callback", "support", "ticket", "service-desk", "incident",
        "call-back", "support callback", "standard-plan"
    ],
    "backup_retention": [
        "backup", "backups", "retention", "database backup", "restore",
        "archive", "backup retention", "production-backup"
    ],
    "deployment": [
        "deployment", "deploy", "release", "reviewer", "sop",
        "health check", "staging", "production deployment", "development-environment"
    ]
}


class FamilyRouter:
    """
    Deterministic lexicon-based router mapping incoming questions to candidate policy families.
    Enforces a strict confidence threshold (default: 0.70) and ambiguity detection to guarantee
    safe fallback to global retrieval without evidence recall degradation.
    """

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold
        self.lexicons = FAMILY_LEXICONS

    def _score_family(self, text_lower: str, tokens: set, family: str, user_context: Optional[Dict[str, Any]] = None) -> float:
        """Computes matching score for a single policy family."""
        score = 0.0
        keywords = self.lexicons.get(family, [])
        for kw in keywords:
            if " " in kw or "-" in kw:
                if kw in text_lower:
                    score += 1.5  # Higher weight for multi-word domain phrases
            else:
                if kw in tokens:
                    score += 1.0

        # Optional boost if explicitly referenced in context conditions
        if user_context:
            ctx_conditions = [str(c).lower() for c in user_context.get("conditions", [])]
            for kw in keywords:
                if any(kw in c for c in ctx_conditions):
                    score += 0.5
                    break

        return score

    def route_query(
        self,
        question_text: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """
        Routes query text to a candidate policy family.
        Returns RoutingResult with candidate_family, confidence, and fallback indicator.
        """
        text_lower = question_text.lower()
        # Tokenize by non-alphanumeric (keeping dashes for terms like 'remote-work')
        tokens = set(re.findall(r"[a-z0-9\-]+", text_lower))

        scores: Dict[str, float] = {}
        for family in self.lexicons:
            score = self._score_family(text_lower, tokens, family, user_context)
            if score > 0:
                scores[family] = score

        if not scores:
            return RoutingResult(
                candidate_family=None,
                confidence=0.0,
                reason="no_keyword_match",
                scores={},
                fallback_to_global=True
            )

        # Sort descending by score
        sorted_families = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_family, top_score = sorted_families[0]

        # Check for ambiguity ties
        if len(sorted_families) > 1:
            second_family, second_score = sorted_families[1]
            if top_score == second_score:
                return RoutingResult(
                    candidate_family=None,
                    confidence=0.50,
                    reason=f"ambiguity_tie_{top_family}_vs_{second_family}",
                    scores=scores,
                    fallback_to_global=True
                )
            
            # Confidence ratio
            conf = top_score / (top_score + second_score)
            if conf < self.confidence_threshold:
                return RoutingResult(
                    candidate_family=None,
                    confidence=round(conf, 3),
                    reason=f"low_confidence_cross_family_{top_family}_and_{second_family}",
                    scores=scores,
                    fallback_to_global=True
                )
            else:
                return RoutingResult(
                    candidate_family=top_family,
                    confidence=round(conf, 3),
                    reason="routed_multi_match_clear_winner",
                    scores=scores,
                    fallback_to_global=False
                )

        # Single clear family matched
        return RoutingResult(
            candidate_family=top_family,
            confidence=1.0,
            reason="single_family_exact_match",
            scores=scores,
            fallback_to_global=False
        )
