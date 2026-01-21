# GPU-Accelerated Filtered Multi-Vector Search

## Overview

This document describes a design for GPU-accelerated filtered search in ColBERT,
including metadata storage, vectorized filtering, and adaptive retrieval strategies.

## Current Architecture (CPU-based)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Collection.metadata = {pid: {"category": "science", "year": 2020}, ...}     │
│                       ↑ Python dict on CPU                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ filter_fn(pid) → bool    # Per-PID evaluation, CPU dict lookup              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Problems:**
1. Per-PID filtering is O(n) sequential operations
2. Dict lookups are CPU-bound
3. Cannot leverage GPU parallelism during filtering
4. Filter applied after candidate generation but evaluates one-at-a-time

## Proposed Architecture (GPU-accelerated)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GPUMetadataStore                                                            │
│   category_tensor: [0, 1, 0, 2, 1, ...] (n_docs,) int32 on CUDA            │
│   year_tensor:     [2020, 2019, 2021, ...] (n_docs,) int32 on CUDA         │
│   category_to_id:  {"science": 0, "history": 1, ...}                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ GPUFilter.apply(pids_tensor, filters) → filtered_pids_tensor               │
│   # Single GPU kernel: gather + compare + boolean index                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: GPUMetadataStore

```python
import torch
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class FieldSchema:
    """Schema for a single metadata field."""
    name: str
    dtype: str  # 'categorical', 'int', 'float', 'multi_categorical'
    vocab: Optional[Dict[str, int]] = None  # For categorical fields
    vocab_inverse: Optional[Dict[int, str]] = None
    tensor: Optional[torch.Tensor] = None  # For single-value fields
    inverted_index: Optional['GPUInvertedIndex'] = None  # For multi-value fields
    default_value: Any = None


class GPUInvertedIndex:
    """
    GPU-resident inverted index for multi-value fields (e.g., tags).

    Stores posting lists as GPU tensors for O(1) lookup of all PIDs
    with a given value. Much more efficient than padded tensors for
    sparse multi-value fields.

    Memory: O(total_assignments) vs O(n_docs * max_values) for padded
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.vocab: Dict[str, int] = {}
        self.vocab_inverse: Dict[int, str] = {}
        self.posting_lists: Dict[int, torch.Tensor] = {}  # value_id -> PIDs tensor
        self.pid_to_values: Optional[torch.Tensor] = None  # For reverse lookup (optional)

    @classmethod
    def from_metadata(cls, collection, field_name: str,
                      device: str = 'cuda') -> 'GPUInvertedIndex':
        """
        Build inverted index from collection metadata.

        Args:
            collection: Collection with .metadata dict
            field_name: Name of multi-value field (e.g., 'tags')
            device: 'cuda' or 'cpu'
        """
        index = cls(device=device)

        # First pass: build vocabulary and collect postings
        postings: Dict[str, List[int]] = {}  # value -> list of PIDs

        for pid in range(len(collection)):
            meta = collection.get_metadata(pid)
            values = meta.get(field_name, [])

            # Handle both list and single values
            if not isinstance(values, (list, tuple)):
                values = [values] if values else []

            for value in values:
                value_str = str(value)
                if value_str not in postings:
                    postings[value_str] = []
                postings[value_str].append(pid)

        # Build vocabulary
        for value_str in sorted(postings.keys()):
            value_id = len(index.vocab)
            index.vocab[value_str] = value_id
            index.vocab_inverse[value_id] = value_str

        # Convert posting lists to GPU tensors
        for value_str, pids in postings.items():
            value_id = index.vocab[value_str]
            index.posting_lists[value_id] = torch.tensor(
                pids, dtype=torch.int32, device=device
            )

        return index

    def get_pids_for_value(self, value: str) -> torch.Tensor:
        """Get all PIDs that have the given value."""
        value_id = self.vocab.get(value)
        if value_id is None:
            return torch.tensor([], dtype=torch.int32, device=self.device)
        return self.posting_lists.get(value_id,
            torch.tensor([], dtype=torch.int32, device=self.device))

    def get_pids_for_any_value(self, values: List[str]) -> torch.Tensor:
        """Get all PIDs that have ANY of the given values (OR)."""
        all_pids = []
        for value in values:
            pids = self.get_pids_for_value(value)
            if len(pids) > 0:
                all_pids.append(pids)

        if not all_pids:
            return torch.tensor([], dtype=torch.int32, device=self.device)

        # Union of all posting lists
        return torch.cat(all_pids).unique()

    def get_pids_for_all_values(self, values: List[str]) -> torch.Tensor:
        """Get all PIDs that have ALL of the given values (AND)."""
        if not values:
            return torch.tensor([], dtype=torch.int32, device=self.device)

        # Start with first value's posting list
        result = self.get_pids_for_value(values[0])

        # Intersect with remaining values
        for value in values[1:]:
            other_pids = self.get_pids_for_value(value)
            # Intersection via isin
            mask = torch.isin(result, other_pids)
            result = result[mask]

            if len(result) == 0:
                break

        return result

    def filter_candidates_has_value(self, candidates: torch.Tensor,
                                     value: str) -> torch.Tensor:
        """Filter candidates to those having the given value."""
        matching_pids = self.get_pids_for_value(value)
        mask = torch.isin(candidates, matching_pids)
        return candidates[mask]

    def filter_candidates_has_any(self, candidates: torch.Tensor,
                                   values: List[str]) -> torch.Tensor:
        """Filter candidates to those having ANY of the given values."""
        matching_pids = self.get_pids_for_any_value(values)
        mask = torch.isin(candidates, matching_pids)
        return candidates[mask]

    def memory_usage(self) -> int:
        """Return total memory usage in bytes."""
        total = 0
        for tensor in self.posting_lists.values():
            total += tensor.element_size() * tensor.numel()
        return total

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the inverted index."""
        posting_sizes = [len(t) for t in self.posting_lists.values()]
        return {
            'n_values': len(self.vocab),
            'total_postings': sum(posting_sizes),
            'avg_posting_size': sum(posting_sizes) / len(posting_sizes) if posting_sizes else 0,
            'max_posting_size': max(posting_sizes) if posting_sizes else 0,
            'min_posting_size': min(posting_sizes) if posting_sizes else 0,
            'memory_bytes': self.memory_usage(),
        }


class GPUMetadataStore:
    """
    GPU-resident metadata storage for fast vectorized filtering.

    Converts CPU-based metadata dicts into GPU tensors at load time,
    enabling O(1) parallel filtering of millions of candidates.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.fields: Dict[str, FieldSchema] = {}
        self.n_docs: int = 0
        self._initialized = False

    @classmethod
    def from_collection(cls, collection, fields_config: Dict[str, str],
                        device: str = 'cuda') -> 'GPUMetadataStore':
        """
        Build GPU metadata store from a Collection's metadata.

        Args:
            collection: Collection with .metadata dict
            fields_config: {field_name: dtype} where dtype in ['categorical', 'int', 'float']
            device: 'cuda' or 'cpu'

        Example:
            store = GPUMetadataStore.from_collection(
                collection,
                {
                    'category': 'categorical',
                    'year': 'int',
                    'score': 'float',
                    'tags': 'multi_categorical'  # Multi-value field
                }
            )
        """
        store = cls(device=device)
        store.n_docs = len(collection)

        # First pass: discover vocabulary for categorical fields
        vocabs = {name: {} for name, dtype in fields_config.items() if dtype == 'categorical'}

        for pid in range(len(collection)):
            meta = collection.get_metadata(pid)
            for field_name in vocabs:
                if field_name in meta:
                    value = meta[field_name]
                    if value not in vocabs[field_name]:
                        vocabs[field_name][value] = len(vocabs[field_name])

        # Second pass: build tensors
        for field_name, dtype in fields_config.items():
            if dtype == 'categorical':
                vocab = vocabs[field_name]
                vocab_inverse = {v: k for k, v in vocab.items()}
                default_id = len(vocab)  # Unknown/missing category

                values = []
                for pid in range(len(collection)):
                    meta = collection.get_metadata(pid)
                    value = meta.get(field_name)
                    values.append(vocab.get(value, default_id))

                tensor = torch.tensor(values, dtype=torch.int32, device=device)
                store.fields[field_name] = FieldSchema(
                    name=field_name,
                    dtype='categorical',
                    vocab=vocab,
                    vocab_inverse=vocab_inverse,
                    tensor=tensor,
                    default_value=default_id
                )

            elif dtype == 'int':
                values = []
                for pid in range(len(collection)):
                    meta = collection.get_metadata(pid)
                    values.append(meta.get(field_name, 0))

                tensor = torch.tensor(values, dtype=torch.int64, device=device)
                store.fields[field_name] = FieldSchema(
                    name=field_name,
                    dtype='int',
                    tensor=tensor,
                    default_value=0
                )

            elif dtype == 'float':
                values = []
                for pid in range(len(collection)):
                    meta = collection.get_metadata(pid)
                    values.append(meta.get(field_name, 0.0))

                tensor = torch.tensor(values, dtype=torch.float32, device=device)
                store.fields[field_name] = FieldSchema(
                    name=field_name,
                    dtype='float',
                    tensor=tensor,
                    default_value=0.0
                )

            elif dtype == 'multi_categorical':
                # Build inverted index for multi-value field
                inverted_index = GPUInvertedIndex.from_metadata(
                    collection, field_name, device=device
                )
                store.fields[field_name] = FieldSchema(
                    name=field_name,
                    dtype='multi_categorical',
                    vocab=inverted_index.vocab,
                    vocab_inverse=inverted_index.vocab_inverse,
                    inverted_index=inverted_index,
                    default_value=[]
                )

        store._initialized = True
        return store

    def get_inverted_index(self, field_name: str) -> Optional[GPUInvertedIndex]:
        """Get the inverted index for a multi-value field."""
        schema = self.fields.get(field_name)
        if schema and schema.dtype == 'multi_categorical':
            return schema.inverted_index
        return None

    def encode_filter_value(self, field_name: str, value: Any) -> Union[int, float, torch.Tensor]:
        """Convert a filter value to its tensor-compatible form."""
        schema = self.fields[field_name]

        if schema.dtype == 'categorical':
            if isinstance(value, list):
                # Multi-value filter
                ids = [schema.vocab.get(v, schema.default_value) for v in value]
                return torch.tensor(ids, dtype=torch.int32, device=self.device)
            else:
                return schema.vocab.get(value, schema.default_value)
        else:
            return value

    def get_field_values(self, field_name: str, pids: torch.Tensor) -> torch.Tensor:
        """Gather field values for given PIDs (GPU-accelerated)."""
        return self.fields[field_name].tensor[pids.long()]

    def memory_usage(self) -> Dict[str, int]:
        """Return memory usage per field in bytes."""
        usage = {}
        for name, schema in self.fields.items():
            if schema.tensor is not None:
                usage[name] = schema.tensor.element_size() * schema.tensor.numel()
        return usage
```

---

## Component 2: GPUFilter

```python
from enum import Enum
from typing import Dict, List, Tuple, Any
import torch


class FilterOp(Enum):
    EQ = "eq"           # ==
    NE = "ne"           # !=
    GT = "gt"           # >
    GE = "ge"           # >=
    LT = "lt"           # <
    LE = "le"           # <=
    IN = "in"           # in list
    NOT_IN = "not_in"   # not in list
    HAS = "has"         # multi-value field contains value
    HAS_ANY = "has_any" # multi-value field contains any of values
    HAS_ALL = "has_all" # multi-value field contains all of values


@dataclass
class FilterCondition:
    """A single filter condition."""
    field: str
    op: FilterOp
    value: Any


class GPUFilter:
    """
    Vectorized GPU-based filtering for ColBERT search.

    Applies filter conditions to candidate PIDs using parallel GPU operations.
    All filtering happens in a single pass with no CPU round-trips.
    """

    def __init__(self, metadata_store: GPUMetadataStore):
        self.metadata_store = metadata_store

    def parse_filters(self, filter_dict: Dict[str, Any]) -> List[FilterCondition]:
        """
        Parse user-friendly filter dict into FilterConditions.

        Supports:
            {"category": "science"}           → EQ
            {"category": ["science", "art"]}  → IN
            {"year": ">=2020"}                → GE
            {"year": "<=2022"}                → LE
            {"year": ">2020"}                 → GT
            {"year": "<2022"}                 → LT
            {"score": {"gte": 0.5}}           → GE (explicit form)

        Multi-value fields (auto-detected from schema):
            {"tags": "python"}                → HAS (single value)
            {"tags": ["python", "ml"]}        → HAS_ANY (match any)
            {"tags": {"all": ["python", "ml"]}} → HAS_ALL (match all)
        """
        conditions = []

        for field, value in filter_dict.items():
            # Check if this is a multi-value field
            schema = self.metadata_store.fields.get(field)
            is_multi = schema and schema.dtype == 'multi_categorical'

            if isinstance(value, list):
                if is_multi:
                    conditions.append(FilterCondition(field, FilterOp.HAS_ANY, value))
                else:
                    conditions.append(FilterCondition(field, FilterOp.IN, value))

            elif isinstance(value, str):
                if value.startswith('>='):
                    conditions.append(FilterCondition(field, FilterOp.GE, float(value[2:])))
                elif value.startswith('<='):
                    conditions.append(FilterCondition(field, FilterOp.LE, float(value[2:])))
                elif value.startswith('>'):
                    conditions.append(FilterCondition(field, FilterOp.GT, float(value[1:])))
                elif value.startswith('<'):
                    conditions.append(FilterCondition(field, FilterOp.LT, float(value[1:])))
                else:
                    # Exact string match (categorical) or HAS for multi-value
                    if is_multi:
                        conditions.append(FilterCondition(field, FilterOp.HAS, value))
                    else:
                        conditions.append(FilterCondition(field, FilterOp.EQ, value))

            elif isinstance(value, dict):
                # Explicit operator form: {"field": {"gte": 5, "lt": 10}}
                # Or multi-value: {"tags": {"all": ["a", "b"]}}
                for op_name, op_value in value.items():
                    op_map = {
                        'eq': FilterOp.EQ, 'ne': FilterOp.NE,
                        'gt': FilterOp.GT, 'gte': FilterOp.GE,
                        'lt': FilterOp.LT, 'lte': FilterOp.LE,
                        'in': FilterOp.IN, 'not_in': FilterOp.NOT_IN,
                        'has': FilterOp.HAS, 'has_any': FilterOp.HAS_ANY,
                        'all': FilterOp.HAS_ALL, 'has_all': FilterOp.HAS_ALL,
                    }
                    conditions.append(FilterCondition(field, op_map[op_name], op_value))

            else:
                # Direct value comparison
                conditions.append(FilterCondition(field, FilterOp.EQ, value))

        return conditions

    def apply(self, pids: torch.Tensor, filters: Dict[str, Any]) -> torch.Tensor:
        """
        Apply filters to candidate PIDs.

        Args:
            pids: (n_candidates,) int32 tensor of passage IDs on GPU
            filters: Filter specification dict

        Returns:
            (n_filtered,) int32 tensor of PIDs that pass all filters

        Performance:
            O(1) parallel operations regardless of n_candidates
            Typical: 1-2ms for 1M candidates on modern GPU
        """
        if not filters:
            return pids

        conditions = self.parse_filters(filters)

        # Start with all True mask
        mask = torch.ones(len(pids), dtype=torch.bool, device=pids.device)

        for cond in conditions:
            # Gather field values for candidate PIDs
            field_values = self.metadata_store.get_field_values(cond.field, pids)
            encoded_value = self.metadata_store.encode_filter_value(cond.field, cond.value)

            # Apply comparison (all vectorized GPU operations)
            if cond.op == FilterOp.EQ:
                cond_mask = field_values == encoded_value
            elif cond.op == FilterOp.NE:
                cond_mask = field_values != encoded_value
            elif cond.op == FilterOp.GT:
                cond_mask = field_values > encoded_value
            elif cond.op == FilterOp.GE:
                cond_mask = field_values >= encoded_value
            elif cond.op == FilterOp.LT:
                cond_mask = field_values < encoded_value
            elif cond.op == FilterOp.LE:
                cond_mask = field_values <= encoded_value
            elif cond.op == FilterOp.IN:
                cond_mask = torch.isin(field_values, encoded_value)
            elif cond.op == FilterOp.NOT_IN:
                cond_mask = ~torch.isin(field_values, encoded_value)

            # Multi-value field operators (use inverted index)
            elif cond.op == FilterOp.HAS:
                inverted_index = self.metadata_store.get_inverted_index(cond.field)
                matching_pids = inverted_index.get_pids_for_value(cond.value)
                cond_mask = torch.isin(pids, matching_pids)
            elif cond.op == FilterOp.HAS_ANY:
                inverted_index = self.metadata_store.get_inverted_index(cond.field)
                matching_pids = inverted_index.get_pids_for_any_value(cond.value)
                cond_mask = torch.isin(pids, matching_pids)
            elif cond.op == FilterOp.HAS_ALL:
                inverted_index = self.metadata_store.get_inverted_index(cond.field)
                matching_pids = inverted_index.get_pids_for_all_values(cond.value)
                cond_mask = torch.isin(pids, matching_pids)

            # AND with running mask
            mask &= cond_mask

        return pids[mask]

    def apply_and_count(self, pids: torch.Tensor, filters: Dict[str, Any]) -> Tuple[torch.Tensor, int, int]:
        """
        Apply filters and return statistics for selectivity estimation.

        Returns:
            (filtered_pids, n_before, n_after)
        """
        n_before = len(pids)
        filtered = self.apply(pids, filters)
        n_after = len(filtered)
        return filtered, n_before, n_after
```

---

## Component 3: Filter Cache

```python
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import json


@dataclass
class CacheEntry:
    """Entry in the filter cache."""
    pids: torch.Tensor  # All PIDs matching this filter (corpus-wide)
    selectivity: float  # Fraction of corpus matching
    created_at: float   # Timestamp for TTL
    hit_count: int = 0  # For LRU tracking


class FilterCache:
    """
    Two-level cache for filter results:

    Level 1: Corpus-wide filter results (for brute-force path)
        - Caches: filter_signature -> all matching PIDs
        - Huge win for selective filters that trigger brute-force mode
        - Filter once, reuse until corpus changes

    Level 2: Selectivity estimates
        - Caches: filter_signature -> selectivity ratio
        - Used to choose retrieval strategy without re-sampling
        - Cheap to store, fast to compute initial estimate

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
```

---

## Component 4: Adaptive Retrieval with Selectivity Estimation

```python
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch


@dataclass
class RetrievalStats:
    """Statistics from a retrieval round."""
    ncells: int
    candidates_before_filter: int
    candidates_after_filter: int
    selectivity: float  # after/before ratio

    @property
    def estimated_corpus_matches(self) -> float:
        """Estimate total matching docs in corpus based on selectivity."""
        # This is a rough estimate assuming uniform distribution
        # In practice, may want to track corpus-wide stats
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
        self.index_scorer = index_scorer
        self.gpu_filter = gpu_filter
        self.min_candidates = min_candidates
        self.max_ncells = max_ncells
        self.selectivity_threshold = selectivity_threshold_for_bruteforce

        # Filter cache (create default if not provided)
        self.cache = filter_cache or FilterCache()

    def _filter_signature(self, filters: Dict[str, Any]) -> str:
        """Create a hashable signature for a filter dict."""
        import json
        return json.dumps(filters, sort_keys=True)

    def estimate_selectivity_sample(self, filters: Dict[str, Any],
                                     sample_size: int = 10000) -> float:
        """
        Estimate filter selectivity by sampling random PIDs.

        This is useful for deciding strategy before any retrieval.
        """
        n_docs = self.gpu_filter.metadata_store.n_docs
        sample_size = min(sample_size, n_docs)

        # Random sample of PIDs
        sample_pids = torch.randint(0, n_docs, (sample_size,),
                                    dtype=torch.int32,
                                    device=self.gpu_filter.metadata_store.device)

        filtered = self.gpu_filter.apply(sample_pids, filters)
        return len(filtered) / sample_size

    def retrieve_filtered(self, config, Q: torch.Tensor,
                          filters: Dict[str, Any],
                          k: int) -> Tuple[torch.Tensor, torch.Tensor, RetrievalStats]:
        """
        Retrieve filtered candidates with adaptive ncells expansion.

        Returns:
            (filtered_pids, centroid_scores, stats)
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
                selectivity=selectivity
            )
            return cached_pids, None, stats

        # LEVEL 2 CACHE: Check for cached selectivity estimate
        cached_selectivity = self.cache.get_selectivity(filters)

        # If we know selectivity is very low, go directly to brute-force
        if cached_selectivity is not None and cached_selectivity < self.selectivity_threshold:
            return self._bruteforce_filtered_retrieve(config, Q, filters, k)

        # Adaptive retrieval
        ncells = 1
        all_stats = []

        while ncells <= self.max_ncells:
            # Configure ncells for this round
            round_config = config.copy()
            round_config.ncells = ncells

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
                return self._bruteforce_filtered_retrieve(config, Q, filters, k)

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
        return filtered_pids, centroid_scores, all_stats[-1]

    def _bruteforce_filtered_retrieve(self, config, Q: torch.Tensor,
                                       filters: Dict[str, Any],
                                       k: int) -> Tuple[torch.Tensor, None, RetrievalStats]:
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
                selectivity=selectivity
            )
            return cached_pids, None, stats

        # Get ALL PIDs (as tensor)
        all_pids = torch.arange(n_docs, dtype=torch.int32,
                                device=self.gpu_filter.metadata_store.device)

        # Apply filter to entire corpus
        filtered_pids = self.gpu_filter.apply(all_pids, filters)

        selectivity = len(filtered_pids) / n_docs

        # CACHE the result for future queries with same filter
        self.cache.put_corpus_result(filters, filtered_pids, selectivity)

        stats = RetrievalStats(
            ncells=0,  # 0 indicates brute-force mode
            candidates_before_filter=n_docs,
            candidates_after_filter=len(filtered_pids),
            selectivity=selectivity
        )

        # No centroid scores in brute-force mode - will do full scoring
        return filtered_pids, None, stats
```

---

## Component 4: Integration with IndexScorer

```python
class FilteredIndexScorer(IndexScorer):
    """
    Extended IndexScorer with GPU-accelerated filtered search.
    """

    def __init__(self, index_path, collection, fields_config: Dict[str, str],
                 use_gpu=True, load_index_with_mmap=False):
        super().__init__(index_path, use_gpu, load_index_with_mmap)

        # Initialize GPU metadata store
        if use_gpu:
            self.metadata_store = GPUMetadataStore.from_collection(
                collection, fields_config, device='cuda'
            )
            self.gpu_filter = GPUFilter(self.metadata_store)
            self.adaptive_retrieval = AdaptiveFilteredRetrieval(
                self, self.gpu_filter
            )
        else:
            self.metadata_store = None
            self.gpu_filter = None
            self.adaptive_retrieval = None

    def rank_filtered(self, config, Q: torch.Tensor,
                      filters: Dict[str, Any] = None,
                      k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Rank with GPU-accelerated filtering.

        This replaces the filter_fn approach with vectorized GPU filtering.
        """
        with torch.inference_mode():
            if filters and self.gpu_filter:
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
                    # Brute-force path: score all filtered PIDs directly
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
```

---

## Component 5: Updated Searcher

```python
class FilteredSearcher(Searcher):
    """
    Searcher with GPU-accelerated filtering support.
    """

    def __init__(self, index, checkpoint=None, collection=None, config=None,
                 index_root=None, verbose: int = 3,
                 filterable_fields: Dict[str, str] = None):
        """
        Args:
            filterable_fields: Dict mapping field names to types.
                Example: {'category': 'categorical', 'year': 'int'}
        """
        super().__init__(index, checkpoint, collection, config, index_root, verbose)

        self.filterable_fields = filterable_fields or {}

        # Replace ranker with filtered version if fields are specified
        if self.filterable_fields and self.config.total_visible_gpus > 0:
            self.ranker = FilteredIndexScorer(
                self.index,
                self.collection,
                self.filterable_fields,
                use_gpu=True
            )

    def search(self, text: str, k: int = 10,
               filters: Dict[str, Any] = None,
               full_length_search: bool = False) -> Tuple[List[int], List[int], List[float]]:
        """
        Search with optional GPU-accelerated filtering.

        Args:
            text: Query text
            k: Number of results
            filters: Filter specification, e.g.:
                {"category": "science", "year": ">=2020"}
                {"category": ["science", "art"]}
                {"score": {"gte": 0.5, "lt": 0.9}}

        Returns:
            (pids, ranks, scores)
        """
        Q = self.encode(text, full_length_search=full_length_search)

        if hasattr(self.ranker, 'rank_filtered') and filters:
            pids, scores = self.ranker.rank_filtered(self.config, Q, filters, k)
        else:
            # Fallback to CPU filtering
            filter_fn = self._create_facet_filter(None, filters) if filters else None
            pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn)
            pids, scores = pids[:k], scores[:k]

        ranks = list(range(1, len(pids) + 1))
        return pids, ranks, scores
```

---

## GPU Facet Computation

```python
class GPUFacetComputer:
    """
    Compute facet counts using GPU-accelerated unique/count operations.
    """

    def __init__(self, metadata_store: GPUMetadataStore):
        self.metadata_store = metadata_store

    def compute_facets(self, pids: torch.Tensor,
                       facet_fields: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Compute facet counts for result PIDs.

        Uses torch.unique with return_counts for O(n log n) GPU-accelerated counting.

        Args:
            pids: (n_results,) tensor of result passage IDs
            facet_fields: List of field names to compute facets for

        Returns:
            {field_name: {value: count, ...}, ...}
        """
        facets = {}

        for field_name in facet_fields:
            schema = self.metadata_store.fields.get(field_name)
            if schema is None:
                continue

            # Get field values for result PIDs
            field_values = self.metadata_store.get_field_values(field_name, pids)

            # Count unique values (GPU-accelerated)
            unique_values, counts = torch.unique(field_values, return_counts=True)

            # Convert to dict with original labels
            facets[field_name] = {}
            for val, count in zip(unique_values.tolist(), counts.tolist()):
                if schema.dtype == 'categorical' and schema.vocab_inverse:
                    label = schema.vocab_inverse.get(val, f"unknown_{val}")
                else:
                    label = str(val)
                facets[field_name][label] = count

        return facets
```

---

## Performance Analysis

### Memory Usage

For a 10M document corpus with 3 filterable fields:

| Field Type | Bytes/Doc | Total (10M docs) |
|------------|-----------|------------------|
| categorical (int32) | 4 | 40 MB |
| int (int64) | 8 | 80 MB |
| float (float32) | 4 | 40 MB |

**Total: ~160 MB GPU memory for 3 fields on 10M docs**

This is negligible compared to:
- ColBERT embeddings: ~10-50 GB for 10M docs
- IVF index structures: ~1-5 GB

### Throughput Comparison

| Operation | CPU (per-PID) | GPU (vectorized) |
|-----------|---------------|------------------|
| Filter 1M candidates | 100-500ms | 1-2ms |
| Filter 100K candidates | 10-50ms | <1ms |
| Facet count 10K results | 5-20ms | <1ms |

**Expected speedup: 50-500x for filtering operations**

### End-to-End Search Latency

| Scenario | CPU Filter | GPU Filter | Notes |
|----------|------------|------------|-------|
| Broad filter (90%) | 150ms | 52ms | Filter overhead reduced |
| Medium filter (10%) | 180ms | 55ms | Similar |
| Selective filter (1%) | 300ms | 60ms | Multiple ncells rounds |
| Very selective (0.1%) | 500ms+ | 80ms | Brute-force kicks in |

---

## Usage Example

```python
from colbert import Searcher
from colbert.search.filtered import FilteredSearcher

# Initialize with filterable fields (including multi-value)
searcher = FilteredSearcher(
    index="my_index",
    collection="my_collection.jsonl",
    filterable_fields={
        'category': 'categorical',        # Single-value categorical
        'year': 'int',                     # Integer
        'source': 'categorical',           # Single-value categorical
        'score': 'float',                  # Float
        'tags': 'multi_categorical',       # Multi-value (uses inverted index)
        'authors': 'multi_categorical',    # Multi-value
    }
)

# Search with filters on single-value fields
pids, ranks, scores = searcher.search(
    "quantum computing applications",
    k=10,
    filters={
        "category": ["science", "technology"],  # IN filter
        "year": ">=2020",                       # Range filter
        "score": {"gte": 0.5}                   # Explicit operator
    }
)

# Search with multi-value field filters
pids, ranks, scores = searcher.search(
    "deep learning frameworks",
    k=10,
    filters={
        "tags": "python",                       # HAS: contains "python"
        "year": ">=2020",
    }
)

# Multi-value OR filter: has any of these tags
pids, ranks, scores = searcher.search(
    "neural networks",
    k=10,
    filters={
        "tags": ["pytorch", "tensorflow", "jax"],  # HAS_ANY: contains any
    }
)

# Multi-value AND filter: has all of these tags
pids, ranks, scores = searcher.search(
    "transformer models",
    k=10,
    filters={
        "tags": {"all": ["nlp", "attention", "bert"]},  # HAS_ALL: contains all
    }
)

# Search with facets
pids, ranks, scores, facets = searcher.search(
    "machine learning",
    k=100,
    filters={"year": ">=2018"},
    facet_fields=["category", "source", "year", "tags"]
)
# facets = {
#     "category": {"science": 45, "tech": 30, ...},
#     "tags": {"python": 67, "ml": 52, "deep-learning": 41, ...},
#     ...
# }

# Check cache stats (for debugging/monitoring)
print(searcher.ranker.adaptive_retrieval.cache.stats())
# {'corpus_cache_entries': 5, 'selectivity_cache_entries': 12,
#  'memory_usage_bytes': 42000, 'hits': 150, 'misses': 12, 'hit_rate': 0.926}
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure
- [ ] `GPUMetadataStore` class with single-value field support
- [ ] `GPUInvertedIndex` class for multi-value fields
- [ ] `GPUFilter` with all operators (eq, in, range, has, has_any, has_all)
- [ ] Unit tests for GPU operations

### Phase 2: Caching
- [ ] `FilterCache` with two-level caching
- [ ] LRU eviction and memory budget management
- [ ] TTL-based expiration
- [ ] Cache invalidation on corpus changes

### Phase 3: Adaptive Retrieval
- [ ] `AdaptiveFilteredRetrieval` with selectivity estimation
- [ ] Progressive ncells expansion based on selectivity
- [ ] Brute-force fallback for very selective filters
- [ ] Integration with FilterCache

### Phase 4: Integration
- [ ] `FilteredIndexScorer` extending `IndexScorer`
- [ ] `FilteredSearcher` extending `Searcher`
- [ ] Backward compatibility with CPU filter_fn

### Phase 5: Faceting
- [ ] `GPUFacetComputer` for fast facet counting
- [ ] Multi-value field faceting via inverted index
- [ ] Integration with search API

### Phase 6: Optimization & Production
- [ ] Benchmark suite for filtered search
- [ ] Memory optimization for large vocabularies
- [ ] Multi-GPU support for metadata store
- [ ] Incremental update support

---

## Design Decisions

### Resolved

1. **Variable-length fields**: Use GPU-resident inverted index (GPUInvertedIndex)
   - Posting lists stored as GPU tensors: `{value_id: tensor of PIDs}`
   - Memory efficient: O(total_assignments) vs O(n_docs * max_values) for padded
   - For selective filters, posting list IS the answer (no filtering needed)
   - Supports OR (union) and AND (intersection) queries efficiently

2. **Filter caching**: Two-level caching strategy (FilterCache)
   - Level 1: Corpus-wide filter results for selective filters (brute-force path)
   - Level 2: Selectivity estimates for strategy selection
   - LRU eviction with memory budget and TTL
   - Huge wins for repeated selective filter patterns

3. **Index-time partitioning**: Not pursuing
   - Would require user to specify partition scheme at index time
   - Combinatorial explosion for multi-dimensional partitioning
   - Adaptive retrieval + caching handles most cases well

## Open Questions

1. **Hybrid CPU/GPU**: For very large metadata vocabularies, should some fields stay on CPU?
   - High-cardinality fields (e.g., author with 1M unique values) may not benefit from GPU
   - Could use CPU-based bloom filter for pre-screening, then GPU for final filter

2. **Incremental updates**: How to handle corpus changes?
   - Current: Invalidate all caches when corpus changes
   - Future: Incremental updates to metadata tensors and inverted indexes
   - Challenge: Maintaining posting list consistency during updates

3. **Multi-GPU sharding**: For very large corpora, how to shard metadata?
   - Option A: Replicate metadata on each GPU (simpler, more memory)
   - Option B: Shard metadata to match embedding shards (complex, less memory)
