"""Unit tests for QueryCache: hashing, LRU eviction, and disk persistence."""

import tempfile
import numpy as np
import pytest
from src.amo.query_cache import QueryCache


def test_query_cache_hit_miss_and_normalization():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = QueryCache(cache_dir=tmp_dir, maxsize=10)

        q1 = "How many remote days per week are allowed?"
        q1_variant = "  HOW MANY  REMOTE DAYS PER WEEK ARE ALLOWED?  "
        dummy_vec = np.random.randn(1024).astype(np.float32)

        # Initially empty -> miss
        assert cache.get(q1) is None
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Store vector
        cache.set(q1, dummy_vec)

        # Retrieval should hit with exact query
        retrieved = cache.get(q1)
        assert retrieved is not None
        np.testing.assert_allclose(retrieved, dummy_vec, rtol=1e-5)

        # Retrieval should hit with whitespace/case variant
        retrieved_var = cache.get(q1_variant)
        assert retrieved_var is not None
        np.testing.assert_allclose(retrieved_var, dummy_vec, rtol=1e-5)

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, 0.01)


def test_query_cache_lru_eviction():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Cache with capacity 2
        cache = QueryCache(cache_dir=tmp_dir, maxsize=2)

        vec1 = np.ones(1024, dtype=np.float32)
        vec2 = np.ones(1024, dtype=np.float32) * 2
        vec3 = np.ones(1024, dtype=np.float32) * 3

        cache.set("query one", vec1)
        cache.set("query two", vec2)

        assert len(cache._memory_cache) == 2
        assert "query one" in [cache.normalize_text("query one")]

        # Access query one to make query two LRU
        _ = cache.get("query one")

        # Insert query three -> should evict query two from memory
        cache.set("query three", vec3)
        assert len(cache._memory_cache) == 2

        k1 = cache.get_cache_key("query one")
        k2 = cache.get_cache_key("query two")
        k3 = cache.get_cache_key("query three")

        assert k1 in cache._memory_cache
        assert k3 in cache._memory_cache
        assert k2 not in cache._memory_cache  # Evicted from memory


def test_query_cache_disk_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache1 = QueryCache(cache_dir=tmp_dir, maxsize=5)
        vec = np.ones(1024, dtype=np.float32) * 42.0

        cache1.set("persistent query test", vec)

        # Create a new cache instance pointing to same directory
        cache2 = QueryCache(cache_dir=tmp_dir, maxsize=5)
        assert len(cache2._memory_cache) == 0  # Cold memory

        # Should load from disk on get
        loaded = cache2.get("persistent query test")
        assert loaded is not None
        np.testing.assert_allclose(loaded, vec, rtol=1e-5)
        assert cache2.hits == 1
