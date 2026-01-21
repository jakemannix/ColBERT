"""
Unit tests for GPU-accelerated filtering components.

Tests can run on CPU for CI environments without GPU.
"""

import pytest
import torch
from unittest.mock import MagicMock

# Determine device - use CPU if CUDA not available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MockCollection:
    """Mock collection for testing."""

    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

    def __len__(self):
        return len(self.data)

    def get_metadata(self, pid):
        return self.metadata.get(pid, {})


@pytest.fixture
def sample_collection():
    """Create a sample collection with metadata."""
    data = [f"Document {i}" for i in range(100)]
    metadata = {
        0: {"category": "science", "year": 2020, "tags": ["python", "ml"]},
        1: {"category": "history", "year": 2019, "tags": ["history"]},
        2: {"category": "science", "year": 2021, "tags": ["python", "data"]},
        3: {"category": "art", "year": 2020, "tags": ["art", "design"]},
        4: {"category": "science", "year": 2022, "tags": ["ml", "deep-learning"]},
        5: {"category": "history", "year": 2018, "tags": ["ancient"]},
        6: {"category": "science", "year": 2020, "tags": ["python"]},
        7: {"category": "tech", "year": 2021, "tags": ["python", "web"]},
        8: {"category": "science", "year": 2023, "tags": ["ml", "nlp", "python"]},
        9: {"category": "art", "year": 2019, "tags": ["painting"]},
    }
    # Fill rest with defaults
    for i in range(10, 100):
        metadata[i] = {
            "category": ["science", "history", "art", "tech"][i % 4],
            "year": 2015 + (i % 10),
            "tags": [f"tag{i % 5}"]
        }
    return MockCollection(data, metadata)


class TestGPUInvertedIndex:
    """Tests for GPUInvertedIndex."""

    def test_build_from_metadata(self, sample_collection):
        from colbert.search.gpu_metadata import GPUInvertedIndex

        index = GPUInvertedIndex.from_metadata(
            sample_collection, "tags", device=DEVICE
        )

        assert len(index.vocab) > 0
        assert "python" in index.vocab
        assert "ml" in index.vocab

    def test_get_pids_for_value(self, sample_collection):
        from colbert.search.gpu_metadata import GPUInvertedIndex

        index = GPUInvertedIndex.from_metadata(
            sample_collection, "tags", device=DEVICE
        )

        # Python should be in docs 0, 2, 6, 7, 8
        python_pids = index.get_pids_for_value("python")
        assert len(python_pids) >= 4
        assert 0 in python_pids.tolist()

    def test_get_pids_for_any_value(self, sample_collection):
        from colbert.search.gpu_metadata import GPUInvertedIndex

        index = GPUInvertedIndex.from_metadata(
            sample_collection, "tags", device=DEVICE
        )

        # OR query: python OR ml
        pids = index.get_pids_for_any_value(["python", "ml"])
        pids_list = pids.tolist()

        # Should include all docs with python or ml
        assert 0 in pids_list  # has both
        assert 4 in pids_list  # has ml

    def test_get_pids_for_all_values(self, sample_collection):
        from colbert.search.gpu_metadata import GPUInvertedIndex

        index = GPUInvertedIndex.from_metadata(
            sample_collection, "tags", device=DEVICE
        )

        # AND query: python AND ml
        pids = index.get_pids_for_all_values(["python", "ml"])
        pids_list = pids.tolist()

        # Should only include docs with BOTH
        assert 0 in pids_list  # has python and ml
        assert 8 in pids_list  # has python and ml

    def test_filter_candidates(self, sample_collection):
        from colbert.search.gpu_metadata import GPUInvertedIndex

        index = GPUInvertedIndex.from_metadata(
            sample_collection, "tags", device=DEVICE
        )

        # Create candidate set
        candidates = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=DEVICE)

        # Filter for python
        filtered = index.filter_candidates_has_value(candidates, "python")
        filtered_list = filtered.tolist()

        assert 0 in filtered_list
        assert 2 in filtered_list
        assert 1 not in filtered_list  # history doc, no python

    def test_empty_result(self, sample_collection):
        from colbert.search.gpu_metadata import GPUInvertedIndex

        index = GPUInvertedIndex.from_metadata(
            sample_collection, "tags", device=DEVICE
        )

        # Query for non-existent tag
        pids = index.get_pids_for_value("nonexistent_tag_xyz")
        assert len(pids) == 0

    def test_memory_usage(self, sample_collection):
        from colbert.search.gpu_metadata import GPUInvertedIndex

        index = GPUInvertedIndex.from_metadata(
            sample_collection, "tags", device=DEVICE
        )

        usage = index.memory_usage()
        assert usage > 0

        stats = index.stats()
        assert stats['n_values'] > 0
        assert stats['total_postings'] > 0


class TestGPUMetadataStore:
    """Tests for GPUMetadataStore."""

    def test_build_from_collection(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {
                'category': 'categorical',
                'year': 'int',
                'tags': 'multi_categorical',
            },
            device=DEVICE
        )

        assert store.n_docs == 100
        assert 'category' in store.fields
        assert 'year' in store.fields
        assert 'tags' in store.fields

    def test_get_field_values(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'year': 'int'},
            device=DEVICE
        )

        pids = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
        years = store.get_field_values('year', pids)

        assert years[0].item() == 2020
        assert years[1].item() == 2019
        assert years[2].item() == 2021

    def test_categorical_encoding(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'category': 'categorical'},
            device=DEVICE
        )

        schema = store.fields['category']
        assert 'science' in schema.vocab
        assert 'history' in schema.vocab

    def test_get_inverted_index(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'tags': 'multi_categorical'},
            device=DEVICE
        )

        index = store.get_inverted_index('tags')
        assert index is not None
        assert "python" in index.vocab


class TestFilterCache:
    """Tests for FilterCache."""

    def test_put_and_get(self):
        from colbert.search.filter_cache import FilterCache

        cache = FilterCache(device=DEVICE)

        filters = {"category": "science"}
        pids = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32, device=DEVICE)

        cache.put_corpus_result(filters, pids, 0.05)

        # Get should return the cached result
        result = cache.get_corpus_result(filters)
        assert result is not None
        assert torch.equal(result, pids)

    def test_cache_miss(self):
        from colbert.search.filter_cache import FilterCache

        cache = FilterCache(device=DEVICE)

        result = cache.get_corpus_result({"category": "nonexistent"})
        assert result is None

    def test_selectivity_cache(self):
        from colbert.search.filter_cache import FilterCache

        cache = FilterCache(device=DEVICE)

        filters = {"category": "science"}
        cache.put_selectivity(filters, 0.15)

        selectivity = cache.get_selectivity(filters)
        assert selectivity == 0.15

    def test_invalidation(self):
        from colbert.search.filter_cache import FilterCache

        cache = FilterCache(device=DEVICE)

        filters = {"category": "science"}
        pids = torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE)
        cache.put_corpus_result(filters, pids, 0.03)

        cache.invalidate()

        result = cache.get_corpus_result(filters)
        assert result is None

    def test_stats(self):
        from colbert.search.filter_cache import FilterCache

        cache = FilterCache(device=DEVICE)

        # Initial stats
        stats = cache.stats()
        assert stats['corpus_cache_entries'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0


class TestGPUFilter:
    """Tests for GPUFilter."""

    def test_parse_filters(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore
        from colbert.search.gpu_filter import GPUFilter, FilterOp

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {
                'category': 'categorical',
                'year': 'int',
                'tags': 'multi_categorical',
            },
            device=DEVICE
        )
        gpu_filter = GPUFilter(store)

        # Test various filter syntaxes
        conditions = gpu_filter.parse_filters({
            "category": "science",       # EQ
            "year": ">=2020",           # GE
            "tags": ["python", "ml"],   # HAS_ANY (multi-value)
        })

        assert len(conditions) == 3

        # Find each condition
        cat_cond = [c for c in conditions if c.field == "category"][0]
        assert cat_cond.op == FilterOp.EQ

        year_cond = [c for c in conditions if c.field == "year"][0]
        assert year_cond.op == FilterOp.GE

        tags_cond = [c for c in conditions if c.field == "tags"][0]
        assert tags_cond.op == FilterOp.HAS_ANY

    def test_apply_eq_filter(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore
        from colbert.search.gpu_filter import GPUFilter

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'category': 'categorical'},
            device=DEVICE
        )
        gpu_filter = GPUFilter(store)

        # All PIDs
        pids = torch.arange(10, dtype=torch.int32, device=DEVICE)

        # Filter for science
        filtered = gpu_filter.apply(pids, {"category": "science"})
        filtered_list = filtered.tolist()

        # PIDs 0, 2, 4, 6, 8 should be science
        assert 0 in filtered_list
        assert 2 in filtered_list
        assert 1 not in filtered_list  # history

    def test_apply_range_filter(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore
        from colbert.search.gpu_filter import GPUFilter

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'year': 'int'},
            device=DEVICE
        )
        gpu_filter = GPUFilter(store)

        pids = torch.arange(10, dtype=torch.int32, device=DEVICE)

        # Filter for year >= 2021
        filtered = gpu_filter.apply(pids, {"year": ">=2021"})
        filtered_list = filtered.tolist()

        # PIDs with year >= 2021: 2 (2021), 4 (2022), 7 (2021), 8 (2023)
        assert 2 in filtered_list
        assert 4 in filtered_list
        assert 0 not in filtered_list  # 2020

    def test_apply_multi_value_filter(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore
        from colbert.search.gpu_filter import GPUFilter

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'tags': 'multi_categorical'},
            device=DEVICE
        )
        gpu_filter = GPUFilter(store)

        pids = torch.arange(10, dtype=torch.int32, device=DEVICE)

        # Filter for docs with python tag
        filtered = gpu_filter.apply(pids, {"tags": "python"})
        filtered_list = filtered.tolist()

        assert 0 in filtered_list  # has python
        assert 2 in filtered_list  # has python
        assert 1 not in filtered_list  # no python

    def test_apply_combined_filters(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore
        from colbert.search.gpu_filter import GPUFilter

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {
                'category': 'categorical',
                'year': 'int',
            },
            device=DEVICE
        )
        gpu_filter = GPUFilter(store)

        pids = torch.arange(10, dtype=torch.int32, device=DEVICE)

        # Filter for science AND year >= 2021
        filtered = gpu_filter.apply(pids, {
            "category": "science",
            "year": ">=2021"
        })
        filtered_list = filtered.tolist()

        # PIDs that are science AND year >= 2021: 2, 4, 8
        assert 2 in filtered_list
        assert 4 in filtered_list
        assert 8 in filtered_list
        assert 0 not in filtered_list  # science but 2020
        assert 7 not in filtered_list  # tech not science

    def test_apply_and_count(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore
        from colbert.search.gpu_filter import GPUFilter

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'category': 'categorical'},
            device=DEVICE
        )
        gpu_filter = GPUFilter(store)

        pids = torch.arange(100, dtype=torch.int32, device=DEVICE)

        filtered, n_before, n_after = gpu_filter.apply_and_count(
            pids, {"category": "science"}
        )

        assert n_before == 100
        assert n_after == len(filtered)
        assert n_after < n_before

    def test_empty_pids(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore
        from colbert.search.gpu_filter import GPUFilter

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'category': 'categorical'},
            device=DEVICE
        )
        gpu_filter = GPUFilter(store)

        pids = torch.tensor([], dtype=torch.int32, device=DEVICE)
        filtered = gpu_filter.apply(pids, {"category": "science"})

        assert len(filtered) == 0

    def test_no_filters(self, sample_collection):
        from colbert.search.gpu_metadata import GPUMetadataStore
        from colbert.search.gpu_filter import GPUFilter

        store = GPUMetadataStore.from_collection(
            sample_collection,
            {'category': 'categorical'},
            device=DEVICE
        )
        gpu_filter = GPUFilter(store)

        pids = torch.arange(10, dtype=torch.int32, device=DEVICE)
        filtered = gpu_filter.apply(pids, {})

        # No filters should return all PIDs
        assert len(filtered) == 10


class TestRetrievalStats:
    """Tests for RetrievalStats."""

    def test_stats_creation(self):
        from colbert.search.adaptive_retrieval import RetrievalStats

        stats = RetrievalStats(
            ncells=4,
            candidates_before_filter=100000,
            candidates_after_filter=5000,
            selectivity=0.05
        )

        assert stats.ncells == 4
        assert stats.selectivity == 0.05
        assert stats.estimated_corpus_selectivity == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
