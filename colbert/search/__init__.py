"""
ColBERT search module.

GPU-accelerated filtered search components:
- GPUMetadataStore: GPU-resident metadata storage
- GPUInvertedIndex: GPU-resident inverted index for multi-value fields
- GPUFilter: Vectorized GPU-based filtering
- FilterCache: Two-level cache for filter results
- AdaptiveFilteredRetrieval: Adaptive retrieval strategy
- FilteredIndexScorer: Extended IndexScorer with filtering support
"""

from .index_storage import IndexScorer
from .gpu_metadata import GPUMetadataStore, GPUInvertedIndex, FieldSchema
from .gpu_filter import GPUFilter, FilterOp, FilterCondition
from .filter_cache import FilterCache, CacheEntry
from .adaptive_retrieval import AdaptiveFilteredRetrieval, RetrievalStats
from .filtered_index_scorer import FilteredIndexScorer, create_filtered_searcher

__all__ = [
    # Core
    'IndexScorer',

    # GPU Metadata
    'GPUMetadataStore',
    'GPUInvertedIndex',
    'FieldSchema',

    # Filtering
    'GPUFilter',
    'FilterOp',
    'FilterCondition',

    # Caching
    'FilterCache',
    'CacheEntry',

    # Adaptive Retrieval
    'AdaptiveFilteredRetrieval',
    'RetrievalStats',

    # Integration
    'FilteredIndexScorer',
    'create_filtered_searcher',
]
