"""
Performance benchmarks for GPU-accelerated filtered search.

Run with: pytest tests/benchmark_gpu_filtering.py -v --benchmark-only

Requires CUDA environment. Tests measure:
1. Metadata store build time and memory usage
2. Inverted index build time and memory usage
3. Filter application latency at various scales
4. Cache hit/miss performance impact
5. Selectivity-based strategy selection
6. End-to-end filtered search comparison

Benchmark configurations:
- Corpus sizes: 100K, 1M, 10M documents
- Filter selectivities: 0.1%, 1%, 10%, 50%, 90%
- Filter types: EQ, IN, range, multi-value (HAS, HAS_ANY, HAS_ALL)
"""

import pytest
import time
import random
import json
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

# Will fail on import if torch not available - that's expected
import torch

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA_AVAILABLE else 'cpu'

# Skip all tests if no CUDA
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


# ============================================================================
# Benchmark Utilities
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    n_docs: int
    n_candidates: int
    selectivity: float
    latency_ms: float
    throughput_ops_per_sec: float
    memory_bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (f"BenchmarkResult({self.name}: {self.latency_ms:.2f}ms, "
                f"selectivity={self.selectivity:.2%}, "
                f"throughput={self.throughput_ops_per_sec:.0f} ops/s)")


class BenchmarkTimer:
    """Context manager for timing with CUDA synchronization."""

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda and CUDA_AVAILABLE
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


def run_benchmark(func, warmup_runs: int = 3, benchmark_runs: int = 10) -> Tuple[float, float]:
    """
    Run a benchmark function multiple times and return (mean_ms, std_ms).

    Includes warmup runs and CUDA synchronization.
    """
    # Warmup
    for _ in range(warmup_runs):
        func()
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        with BenchmarkTimer() as timer:
            func()
        times.append(timer.elapsed_ms)

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ============================================================================
# Synthetic Data Generation
# ============================================================================

class SyntheticCollection:
    """
    Synthetic collection for benchmarking.

    Generates documents with controllable metadata distributions:
    - category: Low cardinality (10-100 values)
    - source: Medium cardinality (100-1000 values)
    - author: High cardinality (1000-100000 values)
    - year: Numeric range (2000-2024)
    - score: Float range (0.0-1.0)
    - tags: Multi-value, variable cardinality
    """

    def __init__(self, n_docs: int,
                 n_categories: int = 50,
                 n_sources: int = 500,
                 n_authors: int = 10000,
                 n_tags: int = 1000,
                 avg_tags_per_doc: float = 3.0,
                 seed: int = 42):
        self.n_docs = n_docs
        self.seed = seed
        random.seed(seed)

        # Generate metadata
        self.metadata = {}
        self.categories = [f"category_{i}" for i in range(n_categories)]
        self.sources = [f"source_{i}" for i in range(n_sources)]
        self.authors = [f"author_{i}" for i in range(n_authors)]
        self.all_tags = [f"tag_{i}" for i in range(n_tags)]

        for pid in range(n_docs):
            # Zipf-like distribution for categories (some categories much more common)
            cat_weights = [1.0 / (i + 1) for i in range(n_categories)]
            category = random.choices(self.categories, weights=cat_weights)[0]

            # Uniform distribution for source
            source = random.choice(self.sources)

            # Zipf-like distribution for authors
            author_weights = [1.0 / (i + 1) ** 0.5 for i in range(n_authors)]
            author = random.choices(self.authors, weights=author_weights)[0]

            # Year: normal distribution centered on 2020
            year = max(2000, min(2024, int(random.gauss(2020, 3))))

            # Score: uniform
            score = random.random()

            # Tags: Poisson-distributed count, Zipf-distributed selection
            n_doc_tags = max(1, int(random.expovariate(1.0 / avg_tags_per_doc)))
            n_doc_tags = min(n_doc_tags, 10)  # Cap at 10 tags
            tag_weights = [1.0 / (i + 1) ** 0.8 for i in range(n_tags)]
            tags = list(set(random.choices(self.all_tags, weights=tag_weights, k=n_doc_tags)))

            self.metadata[pid] = {
                'category': category,
                'source': source,
                'author': author,
                'year': year,
                'score': score,
                'tags': tags,
            }

        # Pre-compute selectivity info
        self._compute_selectivity_stats()

    def _compute_selectivity_stats(self):
        """Compute selectivity statistics for common filters."""
        self.selectivity_stats = {}

        # Category counts
        cat_counts = {}
        for m in self.metadata.values():
            cat = m['category']
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        self.selectivity_stats['category'] = {
            k: v / self.n_docs for k, v in cat_counts.items()
        }

        # Tag counts
        tag_counts = {}
        for m in self.metadata.values():
            for tag in m['tags']:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        self.selectivity_stats['tags'] = {
            k: v / self.n_docs for k, v in tag_counts.items()
        }

    def __len__(self):
        return self.n_docs

    def get_metadata(self, pid):
        return self.metadata.get(pid, {})

    def get_filter_with_selectivity(self, target_selectivity: float,
                                     field: str = 'category') -> Dict[str, Any]:
        """Get a filter that approximately matches target selectivity."""
        if field == 'category':
            # Find category closest to target selectivity
            stats = self.selectivity_stats['category']
            best_cat = min(stats.keys(), key=lambda k: abs(stats[k] - target_selectivity))
            return {'category': best_cat}

        elif field == 'year':
            # Use range filter
            if target_selectivity < 0.1:
                return {'year': '>=2024'}  # Very few docs
            elif target_selectivity < 0.3:
                return {'year': '>=2022'}
            elif target_selectivity < 0.5:
                return {'year': '>=2020'}
            else:
                return {'year': '>=2015'}

        elif field == 'tags':
            # Find tag closest to target selectivity
            stats = self.selectivity_stats['tags']
            best_tag = min(stats.keys(), key=lambda k: abs(stats[k] - target_selectivity))
            return {'tags': best_tag}

        else:
            return {}


# ============================================================================
# Benchmark Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def collection_100k():
    """100K document collection."""
    return SyntheticCollection(n_docs=100_000)


@pytest.fixture(scope="module")
def collection_1m():
    """1M document collection."""
    return SyntheticCollection(n_docs=1_000_000)


@pytest.fixture(scope="module")
def collection_10m():
    """10M document collection (use sparingly - slow to generate)."""
    return SyntheticCollection(n_docs=10_000_000, n_authors=100000, n_tags=5000)


@pytest.fixture(scope="module")
def metadata_store_100k(collection_100k):
    """GPUMetadataStore for 100K docs."""
    from colbert.search.gpu_metadata import GPUMetadataStore
    return GPUMetadataStore.from_collection(
        collection_100k,
        {
            'category': 'categorical',
            'source': 'categorical',
            'year': 'int',
            'score': 'float',
            'tags': 'multi_categorical',
        },
        device=DEVICE
    )


@pytest.fixture(scope="module")
def metadata_store_1m(collection_1m):
    """GPUMetadataStore for 1M docs."""
    from colbert.search.gpu_metadata import GPUMetadataStore
    return GPUMetadataStore.from_collection(
        collection_1m,
        {
            'category': 'categorical',
            'source': 'categorical',
            'year': 'int',
            'score': 'float',
            'tags': 'multi_categorical',
        },
        device=DEVICE
    )


@pytest.fixture(scope="module")
def gpu_filter_100k(metadata_store_100k):
    """GPUFilter for 100K docs."""
    from colbert.search.gpu_filter import GPUFilter
    return GPUFilter(metadata_store_100k)


@pytest.fixture(scope="module")
def gpu_filter_1m(metadata_store_1m):
    """GPUFilter for 1M docs."""
    from colbert.search.gpu_filter import GPUFilter
    return GPUFilter(metadata_store_1m)


# ============================================================================
# Build Time Benchmarks
# ============================================================================

class TestBuildPerformance:
    """Benchmarks for building metadata structures."""

    @pytest.mark.parametrize("n_docs", [10_000, 100_000, 500_000, 1_000_000])
    def test_metadata_store_build_time(self, n_docs):
        """Measure GPUMetadataStore build time at various scales."""
        from colbert.search.gpu_metadata import GPUMetadataStore

        collection = SyntheticCollection(n_docs=n_docs)

        mean_ms, std_ms = run_benchmark(
            lambda: GPUMetadataStore.from_collection(
                collection,
                {
                    'category': 'categorical',
                    'year': 'int',
                    'tags': 'multi_categorical',
                },
                device=DEVICE
            ),
            warmup_runs=1,
            benchmark_runs=3
        )

        # Build the store one more time to get memory stats
        store = GPUMetadataStore.from_collection(
            collection,
            {'category': 'categorical', 'year': 'int', 'tags': 'multi_categorical'},
            device=DEVICE
        )
        memory_bytes = store.total_memory_usage()

        print(f"\n[BUILD] GPUMetadataStore ({n_docs:,} docs):")
        print(f"  Time: {mean_ms:.1f}ms ± {std_ms:.1f}ms")
        print(f"  Memory: {memory_bytes / 1024 / 1024:.1f} MB")
        print(f"  Throughput: {n_docs / mean_ms * 1000:.0f} docs/sec")

    @pytest.mark.parametrize("n_docs,n_tags,avg_tags", [
        (100_000, 100, 2.0),
        (100_000, 1000, 3.0),
        (100_000, 10000, 5.0),
        (1_000_000, 1000, 3.0),
    ])
    def test_inverted_index_build_time(self, n_docs, n_tags, avg_tags):
        """Measure GPUInvertedIndex build time with varying tag cardinality."""
        from colbert.search.gpu_metadata import GPUInvertedIndex

        collection = SyntheticCollection(
            n_docs=n_docs, n_tags=n_tags, avg_tags_per_doc=avg_tags
        )

        mean_ms, std_ms = run_benchmark(
            lambda: GPUInvertedIndex.from_metadata(collection, 'tags', device=DEVICE),
            warmup_runs=1,
            benchmark_runs=3
        )

        # Build one more time for stats
        index = GPUInvertedIndex.from_metadata(collection, 'tags', device=DEVICE)
        stats = index.stats()

        print(f"\n[BUILD] GPUInvertedIndex ({n_docs:,} docs, {n_tags} tags, {avg_tags} avg):")
        print(f"  Time: {mean_ms:.1f}ms ± {std_ms:.1f}ms")
        print(f"  Unique values: {stats['n_values']}")
        print(f"  Total postings: {stats['total_postings']:,}")
        print(f"  Memory: {stats['memory_bytes'] / 1024 / 1024:.1f} MB")


# ============================================================================
# Filter Application Benchmarks
# ============================================================================

class TestFilterPerformance:
    """Benchmarks for filter application."""

    @pytest.mark.parametrize("n_candidates", [10_000, 100_000, 500_000, 1_000_000])
    def test_eq_filter_scaling(self, metadata_store_1m, n_candidates):
        """Measure EQ filter performance at various candidate counts."""
        from colbert.search.gpu_filter import GPUFilter

        gpu_filter = GPUFilter(metadata_store_1m)
        pids = torch.randint(0, metadata_store_1m.n_docs, (n_candidates,),
                             dtype=torch.int32, device=DEVICE)
        filters = {"category": "category_0"}

        mean_ms, std_ms = run_benchmark(
            lambda: gpu_filter.apply(pids, filters),
            warmup_runs=5,
            benchmark_runs=20
        )

        # Get result size
        result = gpu_filter.apply(pids, filters)
        selectivity = len(result) / n_candidates

        print(f"\n[FILTER] EQ filter ({n_candidates:,} candidates):")
        print(f"  Latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        print(f"  Selectivity: {selectivity:.2%}")
        print(f"  Throughput: {n_candidates / mean_ms * 1000 / 1e6:.2f}M candidates/sec")

    @pytest.mark.parametrize("selectivity_target", [0.001, 0.01, 0.1, 0.5, 0.9])
    def test_filter_by_selectivity(self, collection_1m, metadata_store_1m, selectivity_target):
        """Measure filter performance across different selectivities."""
        from colbert.search.gpu_filter import GPUFilter

        gpu_filter = GPUFilter(metadata_store_1m)
        n_candidates = 500_000
        pids = torch.randint(0, metadata_store_1m.n_docs, (n_candidates,),
                             dtype=torch.int32, device=DEVICE)

        # Get filter with approximate target selectivity
        filters = collection_1m.get_filter_with_selectivity(selectivity_target)

        mean_ms, std_ms = run_benchmark(
            lambda: gpu_filter.apply(pids, filters),
            warmup_runs=5,
            benchmark_runs=20
        )

        result = gpu_filter.apply(pids, filters)
        actual_selectivity = len(result) / n_candidates

        print(f"\n[FILTER] Selectivity ~{selectivity_target:.1%} (actual {actual_selectivity:.2%}):")
        print(f"  Filter: {filters}")
        print(f"  Latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        print(f"  Result size: {len(result):,}")

    @pytest.mark.parametrize("filter_type,filter_spec", [
        ("EQ", {"category": "category_0"}),
        ("IN_small", {"category": ["category_0", "category_1"]}),
        ("IN_large", {"category": [f"category_{i}" for i in range(10)]}),
        ("RANGE_GE", {"year": ">=2022"}),
        ("RANGE_LE", {"year": "<=2015"}),
        ("COMBINED", {"category": "category_0", "year": ">=2020"}),
    ])
    def test_filter_type_comparison(self, metadata_store_1m, filter_type, filter_spec):
        """Compare performance of different filter types."""
        from colbert.search.gpu_filter import GPUFilter

        gpu_filter = GPUFilter(metadata_store_1m)
        n_candidates = 500_000
        pids = torch.randint(0, metadata_store_1m.n_docs, (n_candidates,),
                             dtype=torch.int32, device=DEVICE)

        mean_ms, std_ms = run_benchmark(
            lambda: gpu_filter.apply(pids, filter_spec),
            warmup_runs=5,
            benchmark_runs=20
        )

        result = gpu_filter.apply(pids, filter_spec)
        selectivity = len(result) / n_candidates

        print(f"\n[FILTER TYPE] {filter_type}:")
        print(f"  Latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        print(f"  Selectivity: {selectivity:.2%}")


# ============================================================================
# Multi-Value Field Benchmarks
# ============================================================================

class TestMultiValuePerformance:
    """Benchmarks for multi-value field operations."""

    @pytest.mark.parametrize("n_candidates", [10_000, 100_000, 500_000])
    def test_has_single_value(self, metadata_store_1m, n_candidates):
        """Measure HAS filter (single tag) performance."""
        from colbert.search.gpu_filter import GPUFilter

        gpu_filter = GPUFilter(metadata_store_1m)
        pids = torch.randint(0, metadata_store_1m.n_docs, (n_candidates,),
                             dtype=torch.int32, device=DEVICE)
        filters = {"tags": "tag_0"}  # Most common tag

        mean_ms, std_ms = run_benchmark(
            lambda: gpu_filter.apply(pids, filters),
            warmup_runs=5,
            benchmark_runs=20
        )

        result = gpu_filter.apply(pids, filters)
        selectivity = len(result) / n_candidates

        print(f"\n[MULTI-VALUE] HAS single ({n_candidates:,} candidates):")
        print(f"  Latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        print(f"  Selectivity: {selectivity:.2%}")

    @pytest.mark.parametrize("n_values", [2, 5, 10, 20])
    def test_has_any_scaling(self, metadata_store_1m, n_values):
        """Measure HAS_ANY filter scaling with number of values."""
        from colbert.search.gpu_filter import GPUFilter

        gpu_filter = GPUFilter(metadata_store_1m)
        n_candidates = 500_000
        pids = torch.randint(0, metadata_store_1m.n_docs, (n_candidates,),
                             dtype=torch.int32, device=DEVICE)

        tags = [f"tag_{i}" for i in range(n_values)]
        filters = {"tags": tags}

        mean_ms, std_ms = run_benchmark(
            lambda: gpu_filter.apply(pids, filters),
            warmup_runs=5,
            benchmark_runs=20
        )

        result = gpu_filter.apply(pids, filters)
        selectivity = len(result) / n_candidates

        print(f"\n[MULTI-VALUE] HAS_ANY ({n_values} tags):")
        print(f"  Latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        print(f"  Selectivity: {selectivity:.2%}")

    @pytest.mark.parametrize("n_values", [2, 3, 4, 5])
    def test_has_all_scaling(self, metadata_store_1m, n_values):
        """Measure HAS_ALL filter scaling with number of values."""
        from colbert.search.gpu_filter import GPUFilter

        gpu_filter = GPUFilter(metadata_store_1m)
        n_candidates = 500_000
        pids = torch.randint(0, metadata_store_1m.n_docs, (n_candidates,),
                             dtype=torch.int32, device=DEVICE)

        # Use common tags for HAS_ALL (otherwise result is empty)
        tags = [f"tag_{i}" for i in range(n_values)]
        filters = {"tags": {"all": tags}}

        mean_ms, std_ms = run_benchmark(
            lambda: gpu_filter.apply(pids, filters),
            warmup_runs=5,
            benchmark_runs=20
        )

        result = gpu_filter.apply(pids, filters)
        selectivity = len(result) / n_candidates

        print(f"\n[MULTI-VALUE] HAS_ALL ({n_values} tags):")
        print(f"  Latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        print(f"  Selectivity: {selectivity:.2%}")
        print(f"  Result count: {len(result)}")


# ============================================================================
# Cache Performance Benchmarks
# ============================================================================

class TestCachePerformance:
    """Benchmarks for filter caching."""

    def test_cache_hit_vs_miss(self, metadata_store_1m):
        """Compare cache hit vs miss latency."""
        from colbert.search.gpu_filter import GPUFilter
        from colbert.search.filter_cache import FilterCache

        gpu_filter = GPUFilter(metadata_store_1m)
        cache = FilterCache(device=DEVICE)

        filters = {"category": "category_0", "year": ">=2020"}

        # First, measure cache miss (compute filter result)
        all_pids = torch.arange(metadata_store_1m.n_docs, dtype=torch.int32, device=DEVICE)

        miss_times = []
        for _ in range(5):
            cache.invalidate()
            with BenchmarkTimer() as timer:
                result = gpu_filter.apply(all_pids, filters)
                cache.put_corpus_result(filters, result, len(result) / len(all_pids))
            miss_times.append(timer.elapsed_ms)

        # Now measure cache hit
        hit_times = []
        for _ in range(20):
            with BenchmarkTimer() as timer:
                cached_result = cache.get_corpus_result(filters)
            hit_times.append(timer.elapsed_ms)

        miss_mean = statistics.mean(miss_times)
        hit_mean = statistics.mean(hit_times)
        speedup = miss_mean / hit_mean if hit_mean > 0 else float('inf')

        print(f"\n[CACHE] Hit vs Miss performance:")
        print(f"  Cache miss (filter + store): {miss_mean:.3f}ms")
        print(f"  Cache hit (lookup only): {hit_mean:.3f}ms")
        print(f"  Speedup: {speedup:.1f}x")

    def test_cache_memory_overhead(self, metadata_store_1m):
        """Measure cache memory usage for different filter counts."""
        from colbert.search.gpu_filter import GPUFilter
        from colbert.search.filter_cache import FilterCache

        gpu_filter = GPUFilter(metadata_store_1m)
        cache = FilterCache(max_entries=100, device=DEVICE)

        all_pids = torch.arange(metadata_store_1m.n_docs, dtype=torch.int32, device=DEVICE)

        # Add filters with varying selectivity
        for i in range(50):
            filters = {"category": f"category_{i}"}
            result = gpu_filter.apply(all_pids, filters)
            cache.put_corpus_result(filters, result, len(result) / len(all_pids))

        stats = cache.stats()
        print(f"\n[CACHE] Memory usage ({stats['corpus_cache_entries']} entries):")
        print(f"  Total memory: {stats['memory_usage_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Per-entry avg: {stats['memory_usage_bytes'] / max(1, stats['corpus_cache_entries']) / 1024:.1f} KB")


# ============================================================================
# Brute-Force vs ANN Comparison
# ============================================================================

class TestBruteForceVsANN:
    """Compare brute-force filtering vs ANN-based retrieval."""

    @pytest.mark.parametrize("selectivity", [0.001, 0.01, 0.1])
    def test_brute_force_filtering(self, collection_1m, metadata_store_1m, selectivity):
        """Measure brute-force corpus filtering at different selectivities."""
        from colbert.search.gpu_filter import GPUFilter

        gpu_filter = GPUFilter(metadata_store_1m)
        all_pids = torch.arange(metadata_store_1m.n_docs, dtype=torch.int32, device=DEVICE)

        # Get filter with target selectivity
        filters = collection_1m.get_filter_with_selectivity(selectivity)

        mean_ms, std_ms = run_benchmark(
            lambda: gpu_filter.apply(all_pids, filters),
            warmup_runs=3,
            benchmark_runs=10
        )

        result = gpu_filter.apply(all_pids, filters)
        actual_selectivity = len(result) / len(all_pids)

        print(f"\n[BRUTE-FORCE] Corpus filter (target {selectivity:.1%}):")
        print(f"  Actual selectivity: {actual_selectivity:.2%}")
        print(f"  Latency: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        print(f"  Result size: {len(result):,}")
        print(f"  Throughput: {len(all_pids) / mean_ms * 1000 / 1e6:.2f}M docs/sec")


# ============================================================================
# Inverted Index Operations
# ============================================================================

class TestInvertedIndexOperations:
    """Benchmarks for inverted index-specific operations."""

    def test_posting_list_lookup(self, metadata_store_1m):
        """Measure posting list lookup latency."""
        inverted_index = metadata_store_1m.get_inverted_index('tags')
        assert inverted_index is not None

        # Lookup common tag
        mean_ms, std_ms = run_benchmark(
            lambda: inverted_index.get_pids_for_value("tag_0"),
            warmup_runs=10,
            benchmark_runs=100
        )

        result = inverted_index.get_pids_for_value("tag_0")
        print(f"\n[INVERTED INDEX] Posting list lookup:")
        print(f"  Latency: {mean_ms * 1000:.1f}μs ± {std_ms * 1000:.1f}μs")
        print(f"  Posting list size: {len(result):,}")

    def test_union_operation(self, metadata_store_1m):
        """Measure posting list union (OR) performance."""
        inverted_index = metadata_store_1m.get_inverted_index('tags')
        assert inverted_index is not None

        tags = [f"tag_{i}" for i in range(10)]

        mean_ms, std_ms = run_benchmark(
            lambda: inverted_index.get_pids_for_any_value(tags),
            warmup_runs=5,
            benchmark_runs=20
        )

        result = inverted_index.get_pids_for_any_value(tags)
        print(f"\n[INVERTED INDEX] Union of 10 posting lists:")
        print(f"  Latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        print(f"  Result size: {len(result):,}")

    def test_intersection_operation(self, metadata_store_1m):
        """Measure posting list intersection (AND) performance."""
        inverted_index = metadata_store_1m.get_inverted_index('tags')
        assert inverted_index is not None

        tags = [f"tag_{i}" for i in range(3)]  # Fewer tags for AND

        mean_ms, std_ms = run_benchmark(
            lambda: inverted_index.get_pids_for_all_values(tags),
            warmup_runs=5,
            benchmark_runs=20
        )

        result = inverted_index.get_pids_for_all_values(tags)
        print(f"\n[INVERTED INDEX] Intersection of 3 posting lists:")
        print(f"  Latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        print(f"  Result size: {len(result):,}")


# ============================================================================
# Memory Benchmarks
# ============================================================================

class TestMemoryUsage:
    """Benchmarks for memory usage."""

    def test_memory_by_field_type(self, collection_1m):
        """Measure memory usage by field type."""
        from colbert.search.gpu_metadata import GPUMetadataStore

        # Build with different field combinations
        configs = [
            ("categorical_only", {'category': 'categorical'}),
            ("int_only", {'year': 'int'}),
            ("float_only", {'score': 'float'}),
            ("multi_only", {'tags': 'multi_categorical'}),
            ("all_fields", {
                'category': 'categorical',
                'source': 'categorical',
                'year': 'int',
                'score': 'float',
                'tags': 'multi_categorical',
            }),
        ]

        print(f"\n[MEMORY] Usage by field type (1M docs):")
        for name, fields_config in configs:
            store = GPUMetadataStore.from_collection(
                collection_1m, fields_config, device=DEVICE
            )
            memory = store.total_memory_usage()
            print(f"  {name}: {memory / 1024 / 1024:.1f} MB")

    def test_memory_scaling_by_docs(self):
        """Measure memory scaling with document count."""
        from colbert.search.gpu_metadata import GPUMetadataStore

        fields_config = {
            'category': 'categorical',
            'year': 'int',
            'tags': 'multi_categorical',
        }

        print(f"\n[MEMORY] Scaling by document count:")
        for n_docs in [10_000, 100_000, 500_000, 1_000_000]:
            collection = SyntheticCollection(n_docs=n_docs)
            store = GPUMetadataStore.from_collection(
                collection, fields_config, device=DEVICE
            )
            memory = store.total_memory_usage()
            per_doc = memory / n_docs
            print(f"  {n_docs:>10,} docs: {memory / 1024 / 1024:>6.1f} MB ({per_doc:.1f} bytes/doc)")


# ============================================================================
# Summary Report
# ============================================================================

class TestSummaryReport:
    """Generate a summary benchmark report."""

    def test_generate_summary(self, collection_1m, metadata_store_1m):
        """Generate comprehensive benchmark summary."""
        from colbert.search.gpu_filter import GPUFilter
        from colbert.search.filter_cache import FilterCache

        gpu_filter = GPUFilter(metadata_store_1m)

        n_candidates = 500_000
        pids = torch.randint(0, metadata_store_1m.n_docs, (n_candidates,),
                             dtype=torch.int32, device=DEVICE)
        all_pids = torch.arange(metadata_store_1m.n_docs, dtype=torch.int32, device=DEVICE)

        results = {}

        # Filter types
        filter_tests = [
            ("EQ", {"category": "category_0"}),
            ("IN (5 values)", {"category": [f"category_{i}" for i in range(5)]}),
            ("RANGE (>=)", {"year": ">=2020"}),
            ("HAS", {"tags": "tag_0"}),
            ("HAS_ANY (5)", {"tags": [f"tag_{i}" for i in range(5)]}),
            ("Combined", {"category": "category_0", "year": ">=2020"}),
        ]

        print("\n" + "=" * 70)
        print("GPU FILTERING BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Corpus size: {metadata_store_1m.n_docs:,} documents")
        print(f"Candidate set: {n_candidates:,} PIDs")
        print(f"Device: {DEVICE}")
        print("-" * 70)
        print(f"{'Filter Type':<25} {'Latency':<15} {'Selectivity':<15} {'Throughput':<15}")
        print("-" * 70)

        for name, filters in filter_tests:
            mean_ms, std_ms = run_benchmark(
                lambda f=filters: gpu_filter.apply(pids, f),
                warmup_runs=5,
                benchmark_runs=20
            )
            result = gpu_filter.apply(pids, filters)
            selectivity = len(result) / n_candidates
            throughput = n_candidates / mean_ms * 1000 / 1e6

            print(f"{name:<25} {mean_ms:.3f}ms ± {std_ms:.2f} {selectivity:>12.1%} {throughput:>11.1f}M/s")
            results[name] = {'latency_ms': mean_ms, 'selectivity': selectivity}

        # Brute-force
        print("-" * 70)
        print("Brute-force corpus filtering (1M docs):")
        for selectivity_target in [0.01, 0.1]:
            filters = collection_1m.get_filter_with_selectivity(selectivity_target)
            mean_ms, _ = run_benchmark(
                lambda f=filters: gpu_filter.apply(all_pids, f),
                warmup_runs=3,
                benchmark_runs=5
            )
            result = gpu_filter.apply(all_pids, filters)
            print(f"  Selectivity ~{selectivity_target:.0%}: {mean_ms:.2f}ms, {len(result):,} results")

        # Cache performance
        print("-" * 70)
        cache = FilterCache(device=DEVICE)
        filters = {"category": "category_0"}

        # Miss
        with BenchmarkTimer() as timer:
            result = gpu_filter.apply(all_pids, filters)
            cache.put_corpus_result(filters, result, len(result) / len(all_pids))
        miss_time = timer.elapsed_ms

        # Hit
        times = []
        for _ in range(20):
            with BenchmarkTimer() as timer:
                _ = cache.get_corpus_result(filters)
            times.append(timer.elapsed_ms)
        hit_time = statistics.mean(times)

        print(f"Cache miss: {miss_time:.2f}ms")
        print(f"Cache hit:  {hit_time:.4f}ms")
        print(f"Speedup:    {miss_time / hit_time:.0f}x")

        print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
