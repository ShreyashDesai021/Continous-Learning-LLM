"""Persistent Deterministic Query Embedding Cache with LRU In-Memory Layer."""

import re
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np


class QueryCache:
    """
    Sub-millisecond query embedding cache backed by persistent storage on D: drive.
    Features:
    1. Deterministic string normalization & MD5 hashing.
    2. Fast in-memory OrderedDict with LRU eviction policy.
    3. Persistent .npy storage on D: drive.
    4. Hit/miss telemetry.
    """

    def __init__(
        self,
        cache_dir: str = "d:/BTech Project Antigravity/storage/query_embedding_cache",
        maxsize: int = 1000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.maxsize = maxsize
        self._memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        
        # Telemetry
        self.hits = 0
        self.misses = 0

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes query text: lowercase, trimmed, collapsed whitespace."""
        s = text.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def get_cache_key(self, text: str) -> str:
        """Generates MD5 hash of normalized query text."""
        normalized = self.normalize_text(text)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def get(self, query_text: str) -> Optional[np.ndarray]:
        """
        Retrieves cached embedding for query.
        Checks memory LRU first, then disk on D: drive.
        """
        key = self.get_cache_key(query_text)

        # 1. In-memory check (LRU update)
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            self.hits += 1
            return self._memory_cache[key]

        # 2. Disk check
        disk_file = self.cache_dir / f"{key}.npy"
        if disk_file.exists():
            try:
                vec = np.load(disk_file)
                self._put_memory(key, vec)
                self.hits += 1
                return vec
            except Exception:
                pass  # Fall through to miss on corrupted file

        self.misses += 1
        return None

    def _put_memory(self, key: str, vec: np.ndarray) -> None:
        """Adds vector to memory cache, enforcing maxsize LRU eviction."""
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        self._memory_cache[key] = vec
        if len(self._memory_cache) > self.maxsize:
            self._memory_cache.popitem(last=False)

    def set(self, query_text: str, embedding: np.ndarray) -> None:
        """Stores embedding in both in-memory LRU and persistent disk storage."""
        key = self.get_cache_key(query_text)
        vec = np.asarray(embedding, dtype=np.float32)

        # Store in memory
        self._put_memory(key, vec)

        # Persist to D: drive
        disk_file = self.cache_dir / f"{key}.npy"
        if not disk_file.exists():
            np.save(disk_file, vec)

    def clear(self, clear_disk: bool = False) -> None:
        """Resets memory cache and telemetry; optionally clears disk."""
        self._memory_cache.clear()
        self.hits = 0
        self.misses = 0
        if clear_disk and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.npy"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Returns hit/miss metrics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_queries": total,
            "hit_rate": hit_rate,
            "memory_entries": len(self._memory_cache)
        }
