"""Snapshot-Aware Context Cache (Query Pattern Memory) with Intent-Aware Compound Keys."""

import json
import hashlib
import re
import os
import uuid
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from src.core.models import RetrievedChunk
from src.cake.temporal_selector import detect_target_date


@dataclass
class CachedCAKEContext:
    """Immutable resolved CAKE context stored in Query Pattern Memory."""
    governing_chunks: List[RetrievedChunk]
    governing_ids: List[str]
    suppressed_ids: List[str]
    future_ids: List[str]
    has_conflict: bool
    policy_family: str
    snapshot_at: str
    intent_signature: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dataclass to JSON-compatible dictionary."""
        return {
            "governing_chunks": [c.model_dump() for c in self.governing_chunks],
            "governing_ids": self.governing_ids,
            "suppressed_ids": self.suppressed_ids,
            "future_ids": self.future_ids,
            "has_conflict": self.has_conflict,
            "policy_family": self.policy_family,
            "snapshot_at": self.snapshot_at,
            "intent_signature": self.intent_signature
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedCAKEContext":
        """Deserializes dictionary back into CachedCAKEContext."""
        chunks = [RetrievedChunk(**c) for c in data["governing_chunks"]]
        return cls(
            governing_chunks=chunks,
            governing_ids=data["governing_ids"],
            suppressed_ids=data["suppressed_ids"],
            future_ids=data["future_ids"],
            has_conflict=data["has_conflict"],
            policy_family=data["policy_family"],
            snapshot_at=data["snapshot_at"],
            intent_signature=data["intent_signature"]
        )


DOMAIN_PREDICATES = {
    "REVIEWERS": ["reviewer", "reviewers", "approval", "approver", "approvals"],
    "HEALTH": ["health", "health check", "healthcheck"],
    "LOGS": ["log", "logs", "logging"],
    "RECORDKEEPING": ["ticket", "record", "recording", "service-desk", "incident"],
    "DEADLINE": ["deadline", "turnaround", "hours", "response time", "sla"],
    "CEILING": ["ceiling", "claim", "reimbursement", "meal allowance", "cost", "limit"],
    "ALLOWANCE": ["days", "allowance", "remote", "wfh", "telecommute", "remote-work", "maximum"],
    "FAQ_INTERPRETATION": ["faq", "clarification", "informal"],
    "NOTICE_AMENDMENT": ["notice", "amendment", "memo", "settle", "disputed"]
}

STOPWORDS = {
    "what", "is", "the", "are", "how", "many", "can", "i", "we", "for", "in", "at", 
    "to", "of", "a", "an", "on", "as", "per", "policy", "tell", "me", "about", 
    "please", "rules", "rule", "does", "do", "apply", "applies", "allowed", "allow", 
    "today", "currently", "now", "current", "working", "work", "which", "and", "or",
    "there", "any", "be", "with", "after", "before", "between", "from", "since", "under"
}


def compute_intent_signature(
    question_text: str,
    snapshot_at: str,
    user_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Computes a 16-character SHA-256 intent signature.
    Encodes:
    1. Temporal target (historical date vs current snapshot).
    2. Geographic scope (Pune vs Mumbai vs General).
    3. Condition scope (Dev vs Prod, Reviewers vs Health Check, Allowance vs Ceiling).
    4. Query mode (single governing vs history sequence vs contradiction comparison).
    5. Canonical domain predicate pattern (invariant under paraphrasing).
    """
    text_lower = question_text.lower()

    # 1. Temporal target
    target_date = detect_target_date(question_text, snapshot_at)
    tau = target_date if target_date is not None else "CURRENT"

    # 2. Scope attributes: Location
    locs = []
    if user_context and "locations" in user_context:
        locs.extend([loc.upper() for loc in user_context["locations"]])
    if "pune" in text_lower and "PUNE" not in locs:
        locs.append("PUNE")
    if "mumbai" in text_lower and "MUMBAI" not in locs:
        locs.append("MUMBAI")
    sigma_loc = ",".join(sorted(locs)) if locs else "GENERAL"

    # Scope attributes: Condition / Predicate
    conds = []
    if "development" in text_lower or "dev" in text_lower:
        conds.append("DEV")
    elif "production" in text_lower or "prod" in text_lower:
        conds.append("PROD")

    for pred_name, kws in DOMAIN_PREDICATES.items():
        if any(kw in text_lower for kw in kws):
            conds.append(pred_name)

    sigma_cond = ",".join(sorted(conds)) if conds else "DEFAULT"

    # 3. Query Mode
    if any(kw in text_lower for kw in ["sequence", "history", "timeline"]):
        mu_mode = "HISTORY_CHAIN"
    elif any(kw in text_lower for kw in ["contradict", "difference", "compare"]):
        mu_mode = "SCOPE_COMPARISON"
    else:
        mu_mode = "GOVERNING_SINGLE"

    # 4. Canonical Intent Pattern (paraphrase invariant)
    if conds:
        rho_pattern = ",".join(sorted(conds))
    else:
        # Fallback to normalized content words only if no domain predicate matched
        clean_words = [w for w in re.findall(r"[a-z0-9]+", text_lower) if w not in STOPWORDS and len(w) > 2]
        rho_pattern = "-".join(sorted(set(clean_words))) if clean_words else "DEFAULT"

    # Combine into deterministic compound signature
    raw_sig = f"{tau}|{sigma_loc}|{sigma_cond}|{mu_mode}|{rho_pattern}"
    return hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]


class ContextCache:
    """
    Snapshot-Aware Query Pattern Memory.
    Keys are strictly 3-tuples: (snapshot_at, policy_family, intent_signature).
    Enforces snapshot isolation and LRU eviction per policy family.
    """

    def __init__(
        self,
        cache_dir: str = "runtime/research/context_cache",
        max_entries_per_family: int = 200
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_family = max_entries_per_family

        # Memory structure: {snapshot_at: {policy_family: OrderedDict[intent_sig, CachedCAKEContext]}}
        self._cache: Dict[str, Dict[str, OrderedDict[str, CachedCAKEContext]]] = {}

        # Telemetry
        self.hits = 0
        self.misses = 0

    def _get_family_cache_file(self, snapshot_at: str, policy_family: str) -> Path:
        """Determines persistent JSON path for a snapshot-family bucket."""
        h = hashlib.md5(snapshot_at.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"ctx_{h}_{policy_family}.json"

    def get(
        self,
        snapshot_at: str,
        policy_family: str,
        question_text: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[CachedCAKEContext]:
        """
        Retrieves cached CAKE context using compound key (snapshot_at, policy_family, intent_signature).
        """
        intent_sig = compute_intent_signature(question_text, snapshot_at, user_context)

        # 1. Check in-memory LRU
        if snapshot_at in self._cache and policy_family in self._cache[snapshot_at]:
            fam_cache = self._cache[snapshot_at][policy_family]
            if intent_sig in fam_cache:
                fam_cache.move_to_end(intent_sig)
                self.hits += 1
                return fam_cache[intent_sig]

        # 2. Check disk persistence
        disk_file = self._get_family_cache_file(snapshot_at, policy_family)
        if disk_file.exists():
            try:
                with open(disk_file, "r", encoding="utf-8") as f:
                    disk_data = json.load(f)
                if intent_sig in disk_data:
                    context = CachedCAKEContext.from_dict(disk_data[intent_sig])
                    self._put_memory(snapshot_at, policy_family, intent_sig, context)
                    self.hits += 1
                    return context
            except Exception:
                pass

        self.misses += 1
        return None

    def _put_memory(
        self,
        snapshot_at: str,
        policy_family: str,
        intent_sig: str,
        context: CachedCAKEContext
    ) -> None:
        """Stores context in memory structure with LRU bounds."""
        if snapshot_at not in self._cache:
            self._cache[snapshot_at] = {}
        if policy_family not in self._cache[snapshot_at]:
            self._cache[snapshot_at][policy_family] = OrderedDict()

        fam_cache = self._cache[snapshot_at][policy_family]
        if intent_sig in fam_cache:
            fam_cache.move_to_end(intent_sig)
        fam_cache[intent_sig] = context

        if len(fam_cache) > self.max_entries_per_family:
            fam_cache.popitem(last=False)

    def set(
        self,
        snapshot_at: str,
        policy_family: str,
        question_text: str,
        context: CachedCAKEContext,
        user_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stores context in both in-memory LRU and persistent JSON file atomically."""
        intent_sig = compute_intent_signature(question_text, snapshot_at, user_context)
        self._put_memory(snapshot_at, policy_family, intent_sig, context)

        # Update disk persistence atomically
        disk_file = self._get_family_cache_file(snapshot_at, policy_family)
        disk_data = {}
        if disk_file.exists():
            try:
                with open(disk_file, "r", encoding="utf-8") as f:
                    disk_data = json.load(f)
            except Exception:
                disk_data = {}

        disk_data[intent_sig] = context.to_dict()
        tmp_file = self.cache_dir / f"{disk_file.stem}_{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(disk_data, f, indent=2)
            os.replace(tmp_file, disk_file)
        except Exception:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    def clear(self, clear_disk: bool = False) -> None:
        """Clears memory cache and resets hit counters; optionally removes disk files."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        if clear_disk and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Returns hit rate telemetry."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_queries": total,
            "hit_rate": hit_rate
        }
