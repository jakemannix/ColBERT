"""
GPU-resident metadata storage for fast vectorized filtering.

This module provides:
- GPUInvertedIndex: GPU-resident posting lists for multi-value fields
- GPUMetadataStore: GPU tensors for single-value fields + inverted indexes for multi-value
"""

import torch
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass


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

    @classmethod
    def from_metadata(cls, collection, field_name: str,
                      device: str = 'cuda') -> 'GPUInvertedIndex':
        """
        Build inverted index from collection metadata.

        Args:
            collection: Collection with .metadata dict or get_metadata method
            field_name: Name of multi-value field (e.g., 'tags')
            device: 'cuda' or 'cpu'
        """
        index = cls(device=device)

        # First pass: build vocabulary and collect postings
        postings: Dict[str, List[int]] = {}  # value -> list of PIDs

        n_docs = len(collection)
        for pid in range(n_docs):
            if hasattr(collection, 'get_metadata'):
                meta = collection.get_metadata(pid)
            else:
                meta = collection.metadata.get(pid, {})

            values = meta.get(field_name, [])

            # Handle both list and single values
            if not isinstance(values, (list, tuple)):
                values = [values] if values else []

            for value in values:
                value_str = str(value)
                if value_str not in postings:
                    postings[value_str] = []
                postings[value_str].append(pid)

        # Build vocabulary (sorted for determinism)
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
        return self.posting_lists.get(
            value_id,
            torch.tensor([], dtype=torch.int32, device=self.device)
        )

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
            if len(result) == 0:
                break
            other_pids = self.get_pids_for_value(value)
            # Intersection via isin
            mask = torch.isin(result, other_pids)
            result = result[mask]

        return result

    def filter_candidates_has_value(self, candidates: torch.Tensor,
                                     value: str) -> torch.Tensor:
        """Filter candidates to those having the given value."""
        matching_pids = self.get_pids_for_value(value)
        if len(matching_pids) == 0:
            return torch.tensor([], dtype=candidates.dtype, device=candidates.device)
        mask = torch.isin(candidates, matching_pids)
        return candidates[mask]

    def filter_candidates_has_any(self, candidates: torch.Tensor,
                                   values: List[str]) -> torch.Tensor:
        """Filter candidates to those having ANY of the given values."""
        matching_pids = self.get_pids_for_any_value(values)
        if len(matching_pids) == 0:
            return torch.tensor([], dtype=candidates.dtype, device=candidates.device)
        mask = torch.isin(candidates, matching_pids)
        return candidates[mask]

    def filter_candidates_has_all(self, candidates: torch.Tensor,
                                   values: List[str]) -> torch.Tensor:
        """Filter candidates to those having ALL of the given values."""
        matching_pids = self.get_pids_for_all_values(values)
        if len(matching_pids) == 0:
            return torch.tensor([], dtype=candidates.dtype, device=candidates.device)
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

    Supports:
    - 'categorical': Single-value categorical fields (stored as int32 tensor)
    - 'int': Integer fields (stored as int64 tensor)
    - 'float': Float fields (stored as float32 tensor)
    - 'multi_categorical': Multi-value fields (stored as GPUInvertedIndex)
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
            collection: Collection with .metadata dict or get_metadata method
            fields_config: {field_name: dtype} where dtype in
                          ['categorical', 'int', 'float', 'multi_categorical']
            device: 'cuda' or 'cpu'

        Example:
            store = GPUMetadataStore.from_collection(
                collection,
                {
                    'category': 'categorical',
                    'year': 'int',
                    'score': 'float',
                    'tags': 'multi_categorical'
                }
            )
        """
        store = cls(device=device)
        store.n_docs = len(collection)

        # Helper to get metadata
        def get_meta(pid):
            if hasattr(collection, 'get_metadata'):
                return collection.get_metadata(pid)
            return collection.metadata.get(pid, {})

        # First pass: discover vocabulary for categorical fields
        vocabs = {name: {} for name, dtype in fields_config.items()
                  if dtype == 'categorical'}

        for pid in range(len(collection)):
            meta = get_meta(pid)
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
                    meta = get_meta(pid)
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
                    meta = get_meta(pid)
                    values.append(int(meta.get(field_name, 0)))

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
                    meta = get_meta(pid)
                    values.append(float(meta.get(field_name, 0.0)))

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

    def get_field_values(self, field_name: str, pids: torch.Tensor) -> torch.Tensor:
        """Gather field values for given PIDs (GPU-accelerated)."""
        schema = self.fields.get(field_name)
        if schema is None or schema.tensor is None:
            raise ValueError(f"Field '{field_name}' not found or is multi-value")
        return schema.tensor[pids.long()]

    def encode_filter_value(self, field_name: str, value: Any) -> Union[int, float, torch.Tensor]:
        """Convert a filter value to its tensor-compatible form."""
        schema = self.fields[field_name]

        if schema.dtype == 'categorical':
            if isinstance(value, list):
                # Multi-value filter (IN operator)
                ids = [schema.vocab.get(v, schema.default_value) for v in value]
                return torch.tensor(ids, dtype=torch.int32, device=self.device)
            else:
                return schema.vocab.get(value, schema.default_value)
        else:
            return value

    def memory_usage(self) -> Dict[str, int]:
        """Return memory usage per field in bytes."""
        usage = {}
        for name, schema in self.fields.items():
            if schema.tensor is not None:
                usage[name] = schema.tensor.element_size() * schema.tensor.numel()
            elif schema.inverted_index is not None:
                usage[name] = schema.inverted_index.memory_usage()
        return usage

    def total_memory_usage(self) -> int:
        """Return total memory usage in bytes."""
        return sum(self.memory_usage().values())

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the metadata store."""
        field_stats = {}
        for name, schema in self.fields.items():
            if schema.dtype == 'multi_categorical' and schema.inverted_index:
                field_stats[name] = schema.inverted_index.stats()
            elif schema.dtype == 'categorical' and schema.vocab:
                field_stats[name] = {'n_categories': len(schema.vocab)}
            else:
                field_stats[name] = {'dtype': schema.dtype}

        return {
            'n_docs': self.n_docs,
            'n_fields': len(self.fields),
            'total_memory_bytes': self.total_memory_usage(),
            'fields': field_stats,
        }
