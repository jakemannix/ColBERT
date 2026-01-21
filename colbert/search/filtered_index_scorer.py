"""
FilteredIndexScorer: GPU-accelerated filtered search for ColBERT.

Extends IndexScorer with GPU-accelerated metadata filtering,
adaptive retrieval, and caching.
"""

import torch
from typing import Dict, List, Tuple, Any, Optional

from .index_storage import IndexScorer
from .gpu_metadata import GPUMetadataStore
from .gpu_filter import GPUFilter
from .filter_cache import FilterCache
from .adaptive_retrieval import AdaptiveFilteredRetrieval


class FilteredIndexScorer(IndexScorer):
    """
    Extended IndexScorer with GPU-accelerated filtered search.

    Usage:
        scorer = FilteredIndexScorer(
            index_path,
            collection,
            fields_config={
                'category': 'categorical',
                'year': 'int',
                'tags': 'multi_categorical',
            },
            use_gpu=True
        )

        pids, scores = scorer.rank_filtered(
            config, Q,
            filters={"category": "science", "year": ">=2020"},
            k=10
        )
    """

    def __init__(self,
                 index_path: str,
                 collection,
                 fields_config: Dict[str, str],
                 use_gpu: bool = True,
                 load_index_with_mmap: bool = False,
                 cache_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            index_path: Path to ColBERT index
            collection: Collection with metadata
            fields_config: {field_name: dtype} where dtype in
                          ['categorical', 'int', 'float', 'multi_categorical']
            use_gpu: Whether to use GPU acceleration
            load_index_with_mmap: Whether to memory-map the index
            cache_config: Optional config for FilterCache
                         {'max_entries': 1000, 'max_memory_bytes': 1e9, 'ttl_seconds': 3600}
        """
        super().__init__(index_path, use_gpu, load_index_with_mmap)

        self.collection = collection
        self.fields_config = fields_config

        # Initialize GPU metadata store
        device = 'cuda' if use_gpu else 'cpu'
        if fields_config:
            self.metadata_store = GPUMetadataStore.from_collection(
                collection, fields_config, device=device
            )
            self.gpu_filter = GPUFilter(self.metadata_store)

            # Initialize filter cache
            cache_config = cache_config or {}
            self.filter_cache = FilterCache(
                max_entries=cache_config.get('max_entries', 1000),
                max_memory_bytes=cache_config.get('max_memory_bytes', 1_000_000_000),
                ttl_seconds=cache_config.get('ttl_seconds', 3600.0),
                device=device
            )

            # Initialize adaptive retrieval
            self.adaptive_retrieval = AdaptiveFilteredRetrieval(
                index_scorer=self,
                gpu_filter=self.gpu_filter,
                filter_cache=self.filter_cache,
                min_candidates=cache_config.get('min_candidates', 256),
                max_ncells=cache_config.get('max_ncells', 32),
                selectivity_threshold_for_bruteforce=cache_config.get(
                    'selectivity_threshold', 0.001
                )
            )
        else:
            self.metadata_store = None
            self.gpu_filter = None
            self.filter_cache = None
            self.adaptive_retrieval = None

    def rank_filtered(self,
                      config,
                      Q: torch.Tensor,
                      filters: Optional[Dict[str, Any]] = None,
                      k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Rank with GPU-accelerated filtering.

        Args:
            config: ColBERT config
            Q: Query embeddings
            filters: Filter specification, e.g.:
                {"category": "science", "year": ">=2020"}
                {"category": ["science", "art"]}
                {"tags": {"all": ["python", "ml"]}}
            k: Number of results to return

        Returns:
            (pids, scores) - Lists of passage IDs and scores
        """
        with torch.inference_mode():
            if filters and self.adaptive_retrieval:
                # Use adaptive filtered retrieval
                pids, centroid_scores, stats = self.adaptive_retrieval.retrieve_filtered(
                    config, Q, filters, k
                )

                if len(pids) == 0:
                    return [], []

                # Score the filtered candidates
                if centroid_scores is not None:
                    # Normal path: use centroid scores for pruning
                    scores, pids = self.score_pids(config, Q, pids, centroid_scores)
                else:
                    # Brute-force/cache path: score all filtered PIDs directly
                    scores, pids = self.score_pids(config, Q, pids, None)
            else:
                # No filters - standard retrieval
                pids, centroid_scores = self.retrieve(config, Q)
                scores, pids = self.score_pids(config, Q, pids, centroid_scores)

            # Sort by score
            scores_sorter = scores.sort(descending=True)
            pids = pids[scores_sorter.indices].tolist()
            scores = scores_sorter.values.tolist()

            return pids[:k], scores[:k]

    def rank(self, config, Q, filter_fn=None, pids=None):
        """
        Override parent rank() to support both legacy filter_fn and new filters.

        For backward compatibility with existing code.
        """
        if filter_fn is None and pids is None:
            # No filtering - use parent implementation
            return super().rank(config, Q, filter_fn=None, pids=None)

        # Legacy path with filter_fn
        return super().rank(config, Q, filter_fn=filter_fn, pids=pids)

    def estimate_selectivity(self, filters: Dict[str, Any],
                              sample_size: int = 10000) -> float:
        """
        Estimate filter selectivity without performing full retrieval.

        Args:
            filters: Filter specification
            sample_size: Number of random PIDs to sample

        Returns:
            Estimated fraction of corpus matching the filter
        """
        if self.gpu_filter is None:
            return 1.0
        return self.gpu_filter.estimate_selectivity(filters, sample_size)

    def invalidate_cache(self):
        """Invalidate all cached filter results."""
        if self.filter_cache:
            self.filter_cache.invalidate()

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        if self.filter_cache:
            return self.filter_cache.stats()
        return {}

    def metadata_stats(self) -> Dict[str, Any]:
        """Return metadata store statistics."""
        if self.metadata_store:
            return self.metadata_store.stats()
        return {}


# Convenience function for creating a filtered searcher
def create_filtered_searcher(index_path: str,
                              collection,
                              fields_config: Dict[str, str],
                              checkpoint: str = None,
                              use_gpu: bool = True) -> 'FilteredSearcher':
    """
    Create a FilteredSearcher with the given configuration.

    This is a convenience function that sets up all components.

    Args:
        index_path: Path to ColBERT index
        collection: Collection with metadata
        fields_config: {field_name: dtype} mapping
        checkpoint: Path to ColBERT checkpoint (optional)
        use_gpu: Whether to use GPU

    Returns:
        FilteredSearcher instance
    """
    # Import here to avoid circular imports
    from colbert.searcher import Searcher

    class FilteredSearcher(Searcher):
        """Searcher with GPU-accelerated filtering support."""

        def __init__(self, index, checkpoint, collection, fields_config, use_gpu):
            super().__init__(
                index=index,
                checkpoint=checkpoint,
                collection=collection,
                verbose=1
            )

            # Replace ranker with filtered version
            self.ranker = FilteredIndexScorer(
                self.index,
                self.collection,
                fields_config,
                use_gpu=use_gpu
            )
            self.fields_config = fields_config

        def search(self, text: str, k: int = 10,
                   filters: Optional[Dict[str, Any]] = None,
                   full_length_search: bool = False) -> Tuple[List[int], List[int], List[float]]:
            """
            Search with optional GPU-accelerated filtering.

            Args:
                text: Query text
                k: Number of results
                filters: Filter specification
                full_length_search: Whether to use full-length search

            Returns:
                (pids, ranks, scores)
            """
            Q = self.encode(text, full_length_search=full_length_search)

            if filters and hasattr(self.ranker, 'rank_filtered'):
                pids, scores = self.ranker.rank_filtered(self.config, Q, filters, k)
            else:
                pids, scores = self.ranker.rank(self.config, Q)
                pids, scores = pids[:k], scores[:k]

            ranks = list(range(1, len(pids) + 1))
            return pids, ranks, scores

    return FilteredSearcher(index_path, checkpoint, collection, fields_config, use_gpu)
