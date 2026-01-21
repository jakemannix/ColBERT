"""
Adaptive retrieval strategy for filtered search.

Adjusts ncells based on filter selectivity and uses caching
for repeated filter patterns.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from .gpu_filter import GPUFilter
from .filter_cache import FilterCache


@dataclass
class RetrievalStats:
    """Statistics from a retrieval round."""
    ncells: int
    candidates_before_filter: int
    candidates_after_filter: int
    selectivity: float  # after/before ratio
    cache_hit: bool = False

    @property
    def estimated_corpus_selectivity(self) -> float:
        """Estimate based on this retrieval round."""
        return self.selectivity


class AdaptiveFilteredRetrieval:
    """
    Adaptive retrieval strategy that adjusts ncells based on filter selectivity.

    Strategy:
    1. Check cache for corpus-wide filter result (instant for selective filters)
    2. If not cached, start with ncells=1
    3. Retrieve candidates, apply GPU filter
    4. Estimate selectivity from filtered/total ratio
    5. If not enough candidates, expand ncells and repeat
    6. Stop when we have enough candidates OR exhausted max_ncells
    7. For very selective filters (<0.1%), switch to brute-force and cache result

    Caching provides huge wins for repeated selective filters.
    """

    def __init__(self,
                 index_scorer,
                 gpu_filter: GPUFilter,
                 filter_cache: Optional[FilterCache] = None,
                 min_candidates: int = 256,
                 max_ncells: int = 32,
                 selectivity_threshold_for_bruteforce: float = 0.001):
        """
        Args:
            index_scorer: The IndexScorer instance for candidate generation
            gpu_filter: GPUFilter instance for applying filters
            filter_cache: Optional FilterCache for caching results
            min_candidates: Minimum number of filtered candidates needed
            max_ncells: Maximum number of centroid cells to probe
            selectivity_threshold_for_bruteforce: Below this selectivity, use brute-force
        """
        self.index_scorer = index_scorer
        self.gpu_filter = gpu_filter
        self.min_candidates = min_candidates
        self.max_ncells = max_ncells
        self.selectivity_threshold = selectivity_threshold_for_bruteforce

        # Filter cache (create default if not provided)
        self.cache = filter_cache or FilterCache()

    def retrieve_filtered(self, config, Q: torch.Tensor,
                          filters: Dict[str, Any],
                          k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], RetrievalStats]:
        """
        Retrieve filtered candidates with adaptive ncells expansion.

        Args:
            config: ColBERT config with ncells, centroid_score_threshold, etc.
            Q: Query embeddings tensor
            filters: Filter specification dict
            k: Number of results requested

        Returns:
            (filtered_pids, centroid_scores, stats)
            centroid_scores is None for cache hits and brute-force mode
        """
        # LEVEL 1 CACHE: Check for cached corpus-wide result
        cached_pids = self.cache.get_corpus_result(filters)
        if cached_pids is not None:
            # Cache hit! Use cached PIDs directly (brute-force path)
            selectivity = len(cached_pids) / self.gpu_filter.metadata_store.n_docs
            stats = RetrievalStats(
                ncells=0,  # 0 indicates cache hit / brute-force
                candidates_before_filter=self.gpu_filter.metadata_store.n_docs,
                candidates_after_filter=len(cached_pids),
                selectivity=selectivity,
                cache_hit=True
            )
            return cached_pids, None, stats

        # LEVEL 2 CACHE: Check for cached selectivity estimate
        cached_selectivity = self.cache.get_selectivity(filters)

        # If we know selectivity is very low, go directly to brute-force
        if cached_selectivity is not None and cached_selectivity < self.selectivity_threshold:
            return self._bruteforce_filtered_retrieve(filters)

        # Adaptive retrieval
        ncells = 1
        all_stats = []
        filtered_pids = None
        centroid_scores = None

        while ncells <= self.max_ncells:
            # Configure ncells for this round
            round_config = self._copy_config_with_ncells(config, ncells)

            # Retrieve candidates (GPU)
            pids, centroid_scores = self.index_scorer.generate_candidates(round_config, Q)

            # Apply GPU filter
            filtered_pids, n_before, n_after = self.gpu_filter.apply_and_count(pids, filters)

            selectivity = n_after / n_before if n_before > 0 else 0.0
            stats = RetrievalStats(ncells, n_before, n_after, selectivity)
            all_stats.append(stats)

            # Update selectivity cache
            self.cache.put_selectivity(filters, selectivity)

            # Check if we have enough candidates
            target = max(self.min_candidates, k * 4)
            if len(filtered_pids) >= target:
                return filtered_pids, centroid_scores, stats

            # Check for very selective filter - switch to brute-force
            if selectivity < self.selectivity_threshold and ncells >= 4:
                return self._bruteforce_filtered_retrieve(filters)

            # Decide expansion factor based on selectivity
            if selectivity < 0.001:
                ncells = min(ncells * 8, self.max_ncells)
            elif selectivity < 0.01:
                ncells = min(ncells * 4, self.max_ncells)
            elif selectivity < 0.1:
                ncells = min(ncells * 2, self.max_ncells)
            else:
                ncells += 1

        # Return what we have after max_ncells
        final_stats = all_stats[-1] if all_stats else RetrievalStats(0, 0, 0, 0.0)
        return filtered_pids if filtered_pids is not None else torch.tensor([], dtype=torch.int32), centroid_scores, final_stats

    def _bruteforce_filtered_retrieve(self, filters: Dict[str, Any]) -> Tuple[torch.Tensor, None, RetrievalStats]:
        """
        For very selective filters: enumerate all matching PIDs and score directly.

        This skips ANN retrieval entirely when the filter is so selective
        that brute-force scoring is faster than probing many centroids.

        Results are cached for future queries with the same filter.
        """
        n_docs = self.gpu_filter.metadata_store.n_docs

        # Check cache first (might have been populated by another query)
        cached_pids = self.cache.get_corpus_result(filters)
        if cached_pids is not None:
            selectivity = len(cached_pids) / n_docs
            stats = RetrievalStats(
                ncells=0,
                candidates_before_filter=n_docs,
                candidates_after_filter=len(cached_pids),
                selectivity=selectivity,
                cache_hit=True
            )
            return cached_pids, None, stats

        # Get ALL PIDs (as tensor)
        all_pids = torch.arange(
            n_docs, dtype=torch.int32,
            device=self.gpu_filter.metadata_store.device
        )

        # Apply filter to entire corpus
        filtered_pids = self.gpu_filter.apply(all_pids, filters)

        selectivity = len(filtered_pids) / n_docs

        # CACHE the result for future queries with same filter
        self.cache.put_corpus_result(filters, filtered_pids, selectivity)

        stats = RetrievalStats(
            ncells=0,  # 0 indicates brute-force mode
            candidates_before_filter=n_docs,
            candidates_after_filter=len(filtered_pids),
            selectivity=selectivity,
            cache_hit=False
        )

        # No centroid scores in brute-force mode - will do full scoring
        return filtered_pids, None, stats

    def _copy_config_with_ncells(self, config, ncells: int):
        """Create a copy of config with modified ncells."""
        # ColBERTConfig is mutable, so we need to be careful
        # For now, we'll create a simple namespace-like object
        class ConfigCopy:
            pass

        copy = ConfigCopy()
        for attr in dir(config):
            if not attr.startswith('_'):
                try:
                    setattr(copy, attr, getattr(config, attr))
                except:
                    pass
        copy.ncells = ncells
        return copy

    def estimate_selectivity_sample(self, filters: Dict[str, Any],
                                     sample_size: int = 10000) -> float:
        """
        Estimate filter selectivity by sampling random PIDs.

        This is useful for deciding strategy before any retrieval.
        """
        return self.gpu_filter.estimate_selectivity(filters, sample_size)

    def invalidate_cache(self):
        """Invalidate all cached filter results."""
        self.cache.invalidate()

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return self.cache.stats()
