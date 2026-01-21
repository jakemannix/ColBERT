"""
Two-level filter cache for GPU-accelerated filtered search.

Level 1: Corpus-wide filter results (for brute-force path)
    - Caches: filter_signature -> all matching PIDs
    - Huge win for selective filters that trigger brute-force mode

Level 2: Selectivity estimates
    - Caches: filter_signature -> selectivity ratio
    - Used to choose retrieval strategy without re-sampling
"""

import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import torch


@dataclass
class CacheEntry:
    """Entry in the filter cache."""
    pids: torch.Tensor  # All PIDs matching this filter (corpus-wide)
    selectivity: float  # Fraction of corpus matching
    created_at: float   # Timestamp for TTL
    hit_count: int = 0  # For LRU tracking


class FilterCache:
    """
    Two-level cache for filter results.

    Memory management:
        - LRU eviction when cache exceeds max_entries
        - TTL expiration for stale entries
        - Memory budget enforcement
    """

    def __init__(self,
                 max_entries: int = 1000,
                 max_memory_bytes: int = 1_000_000_000,  # 1GB default
                 ttl_seconds: float = 3600.0,  # 1 hour default
                 device: str = 'cuda'):
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.ttl_seconds = ttl_seconds
        self.device = device

        # Level 1: Corpus-wide filter results
        self._corpus_cache: Dict[str, CacheEntry] = {}

        # Level 2: Selectivity estimates (cheap, keep more)
        self._selectivity_cache: Dict[str, float] = {}

        # Stats
        self.hits = 0
        self.misses = 0

    def _filter_signature(self, filters: Dict[str, Any]) -> str:
        """Create hashable signature for filter dict."""
        return json.dumps(filters, sort_keys=True)

    def get_corpus_result(self, filters: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Get cached corpus-wide filter result.

        Returns:
            Tensor of all PIDs matching filter, or None if not cached
        """
        sig = self._filter_signature(filters)
        entry = self._corpus_cache.get(sig)

        if entry is None:
            self.misses += 1
            return None

        # Check TTL
        if time.time() - entry.created_at > self.ttl_seconds:
            del self._corpus_cache[sig]
            self.misses += 1
            return None

        # Cache hit
        entry.hit_count += 1
        self.hits += 1
        return entry.pids

    def put_corpus_result(self, filters: Dict[str, Any],
                          pids: torch.Tensor,
                          selectivity: float):
        """
        Cache corpus-wide filter result.

        Args:
            filters: The filter specification
            pids: All PIDs matching the filter
            selectivity: Fraction of corpus matching (for stats)
        """
        sig = self._filter_signature(filters)

        # Check memory budget before adding
        entry_size = pids.element_size() * pids.numel()
        if entry_size > self.max_memory_bytes * 0.1:
            # Single entry too large (>10% of budget), don't cache
            return

        self._evict_if_needed(entry_size)

        self._corpus_cache[sig] = CacheEntry(
            pids=pids,
            selectivity=selectivity,
            created_at=time.time(),
            hit_count=0
        )

        # Also update selectivity cache
        self._selectivity_cache[sig] = selectivity

    def get_selectivity(self, filters: Dict[str, Any]) -> Optional[float]:
        """Get cached selectivity estimate."""
        sig = self._filter_signature(filters)
        return self._selectivity_cache.get(sig)

    def put_selectivity(self, filters: Dict[str, Any], selectivity: float):
        """Cache selectivity estimate."""
        sig = self._filter_signature(filters)
        self._selectivity_cache[sig] = selectivity

    def _evict_if_needed(self, needed_bytes: int):
        """Evict entries if over memory or count budget."""
        # Check entry count
        while len(self._corpus_cache) >= self.max_entries:
            self._evict_lru()

        # Check memory budget
        current_memory = self._current_memory_usage()
        while current_memory + needed_bytes > self.max_memory_bytes:
            if not self._evict_lru():
                break  # Nothing left to evict
            current_memory = self._current_memory_usage()

    def _evict_lru(self) -> bool:
        """Evict least recently used entry. Returns True if evicted."""
        if not self._corpus_cache:
            return False

        # Find LRU entry (lowest hit_count, oldest)
        lru_sig = min(
            self._corpus_cache.keys(),
            key=lambda s: (self._corpus_cache[s].hit_count,
                          -self._corpus_cache[s].created_at)
        )
        del self._corpus_cache[lru_sig]
        return True

    def _current_memory_usage(self) -> int:
        """Calculate current memory usage in bytes."""
        total = 0
        for entry in self._corpus_cache.values():
            total += entry.pids.element_size() * entry.pids.numel()
        return total

    def invalidate(self):
        """Invalidate all cache entries (e.g., when corpus changes)."""
        self._corpus_cache.clear()
        self._selectivity_cache.clear()
        self.hits = 0
        self.misses = 0

    def invalidate_filter(self, filters: Dict[str, Any]):
        """Invalidate a specific filter's cache entries."""
        sig = self._filter_signature(filters)
        if sig in self._corpus_cache:
            del self._corpus_cache[sig]
        if sig in self._selectivity_cache:
            del self._selectivity_cache[sig]

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            'corpus_cache_entries': len(self._corpus_cache),
            'selectivity_cache_entries': len(self._selectivity_cache),
            'memory_usage_bytes': self._current_memory_usage(),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (f"FilterCache(entries={stats['corpus_cache_entries']}, "
                f"memory={stats['memory_usage_bytes']/1024/1024:.1f}MB, "
                f"hit_rate={stats['hit_rate']:.1%})")
