"""
Inverted-list filter index for in-line filtered late-interaction search.

This module provides:

  - ``FilterIndex``       : maps ``(field, value) -> sorted PID list`` via a
                            ``StridedTensor`` over int32 PIDs.
  - ``FilterExpression``  : logical query (``Term`` / ``And`` / ``Or``).
  - ``MaterializedFilter``: query-time representation, either ``BitmapFilter``
                            or ``SortedListFilter``. Both expose
                            ``filter_packed(packed_pids, lengths)``, which is
                            the primitive invoked while walking the IVF cell
                            lists during candidate generation.

The materialised filter is computed once per query and reused across every
centroid cell touched by that query, so the per-query setup cost amortises over
``ncells * query_maxlen`` per-cell intersections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from colbert.search.strided_tensor import StridedTensor


PID_DTYPE = torch.int32


def _to_pid_tensor(values: Iterable[int], device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        t = values.to(dtype=PID_DTYPE)
    else:
        t = torch.tensor(list(values), dtype=PID_DTYPE)
    if device is not None:
        t = t.to(device)
    return t


class FilterIndex:
    """Inverted lists keyed by ``(field, value)``.

    A passage may be associated with multiple values per field (multi-valued
    fields are supported by passing ``Sequence[Any]`` as the per-passage value).
    Posting lists are stored sorted-ascending and packed into a single
    ``StridedTensor`` for cheap term lookup.
    """

    def __init__(
        self,
        vocab: Dict[Tuple[str, Any], int],
        posting_lists: StridedTensor,
        num_passages: int,
    ):
        self._vocab = vocab
        self._posting_lists = posting_lists
        self._num_passages = int(num_passages)

    @property
    def num_passages(self) -> int:
        return self._num_passages

    @property
    def num_terms(self) -> int:
        return len(self._vocab)

    @property
    def fields(self) -> List[str]:
        return sorted({f for f, _ in self._vocab.keys()})

    def has_term(self, field: str, value: Any) -> bool:
        return (field, value) in self._vocab

    def term_id(self, field: str, value: Any) -> Optional[int]:
        return self._vocab.get((field, value))

    def term_pids(self, field: str, value: Any) -> torch.Tensor:
        """Sorted-ascending int32 tensor of PIDs that match ``field == value``.

        Returns an empty tensor if the term is unknown.
        """
        tid = self._vocab.get((field, value))
        if tid is None:
            return torch.empty(0, dtype=PID_DTYPE)
        pids, _ = self._posting_lists.lookup(torch.tensor([tid], dtype=torch.long))
        return pids.to(PID_DTYPE)

    @classmethod
    def from_field_data(
        cls,
        field_data: Dict[str, Sequence[Any]],
        num_passages: int,
        use_gpu: bool = False,
    ) -> "FilterIndex":
        """Build an index from a dict of ``field_name -> per-passage value(s)``.

        Each entry may be a flat sequence (single-valued field) or a sequence
        of sequences (multi-valued: ``per_pid_values[pid] -> Sequence[value]``).
        Missing values are skipped — if a passage has no entry for a field, it
        is simply absent from that field's posting lists.
        """
        vocab: Dict[Tuple[str, Any], int] = {}
        term_to_pids: Dict[int, List[int]] = {}

        for field, per_pid in field_data.items():
            if len(per_pid) > num_passages:
                raise ValueError(
                    f"field {field!r} has {len(per_pid)} entries but num_passages={num_passages}"
                )
            for pid, raw in enumerate(per_pid):
                if raw is None:
                    continue
                values: Sequence[Any]
                if isinstance(raw, (list, tuple, set, frozenset)):
                    values = list(raw)
                else:
                    values = [raw]
                for v in values:
                    key = (field, v)
                    tid = vocab.get(key)
                    if tid is None:
                        tid = len(vocab)
                        vocab[key] = tid
                        term_to_pids[tid] = []
                    term_to_pids[tid].append(pid)

        if len(vocab) == 0:
            packed = torch.empty(0, dtype=PID_DTYPE)
            lengths = torch.empty(0, dtype=torch.long)
        else:
            packed_parts: List[torch.Tensor] = []
            lengths_list: List[int] = []
            for tid in range(len(vocab)):
                pids_for_term = sorted(set(term_to_pids[tid]))
                packed_parts.append(_to_pid_tensor(pids_for_term))
                lengths_list.append(len(pids_for_term))
            packed = torch.cat(packed_parts) if packed_parts else torch.empty(0, dtype=PID_DTYPE)
            lengths = torch.tensor(lengths_list, dtype=torch.long)

        posting_lists = StridedTensor(packed, lengths, use_gpu=use_gpu)
        return cls(vocab=vocab, posting_lists=posting_lists, num_passages=num_passages)


# ---------------------------------------------------------------------------
# Query expressions
# ---------------------------------------------------------------------------


class FilterExpression(ABC):
    """Logical filter expression. Compose with ``Term``, ``And``, ``Or``."""

    @abstractmethod
    def _collect_sorted_pids(self, fi: FilterIndex) -> torch.Tensor:
        """Internal: return the (sorted-ascending) allowed-PID tensor."""

    def materialize(
        self,
        filter_index: FilterIndex,
        mode: str = "auto",
        device: Optional[torch.device] = None,
        bitmap_density_threshold: float = 0.02,
    ) -> "MaterializedFilter":
        """Compile the expression into a runtime ``MaterializedFilter``.

        ``mode``:
          - ``'sorted'``: always return a ``SortedListFilter``.
          - ``'bitmap'``: always return a ``BitmapFilter``.
          - ``'auto'``  : pick by density. Bitmap memory is ``O(N_passages)``,
                          sorted memory is ``O(N_allowed)``. Bitmap also gives
                          branch-free O(1) membership, which is much friendlier
                          for GPU. So we pick ``BitmapFilter`` whenever the
                          allowed set is at least ``bitmap_density_threshold``
                          of the collection, OR whenever target device is GPU.
        """
        allowed = self._collect_sorted_pids(filter_index)
        if device is not None:
            allowed = allowed.to(device)
        N = filter_index.num_passages
        chosen = mode
        if chosen == "auto":
            on_gpu = (device is not None) and (torch.device(device).type == "cuda")
            density = allowed.numel() / max(N, 1)
            chosen = "bitmap" if (on_gpu or density >= bitmap_density_threshold) else "sorted"
        if chosen == "bitmap":
            return BitmapFilter.from_sorted_pids(allowed, N)
        if chosen == "sorted":
            return SortedListFilter(allowed, N)
        raise ValueError(f"unknown materialization mode: {mode!r}")


class Term(FilterExpression):
    def __init__(self, field: str, value: Any):
        self.field = field
        self.value = value

    def _collect_sorted_pids(self, fi: FilterIndex) -> torch.Tensor:
        return fi.term_pids(self.field, self.value)

    def __repr__(self) -> str:
        return f"Term({self.field!r}, {self.value!r})"


class Or(FilterExpression):
    def __init__(self, *children: FilterExpression):
        if not children:
            raise ValueError("Or requires at least one child")
        self.children = list(children)

    def _collect_sorted_pids(self, fi: FilterIndex) -> torch.Tensor:
        parts = [c._collect_sorted_pids(fi) for c in self.children]
        if not parts:
            return torch.empty(0, dtype=PID_DTYPE)
        cat = torch.cat(parts)
        # torch.unique returns sorted-ascending unique values
        return torch.unique(cat).to(PID_DTYPE)

    def __repr__(self) -> str:
        return "Or(" + ", ".join(repr(c) for c in self.children) + ")"


class And(FilterExpression):
    def __init__(self, *children: FilterExpression):
        if not children:
            raise ValueError("And requires at least one child")
        self.children = list(children)

    def _collect_sorted_pids(self, fi: FilterIndex) -> torch.Tensor:
        parts = [c._collect_sorted_pids(fi) for c in self.children]
        # Intersect: sort by length and progressively reduce
        parts.sort(key=lambda t: t.numel())
        result = parts[0]
        for p in parts[1:]:
            if result.numel() == 0:
                break
            # both sorted-ascending; use searchsorted for vectorized membership
            idx = torch.searchsorted(p, result)
            idx = idx.clamp(max=p.numel() - 1) if p.numel() > 0 else idx
            if p.numel() == 0:
                result = result.new_empty(0)
                break
            mask = p[idx] == result
            result = result[mask]
        return result.to(PID_DTYPE)

    def __repr__(self) -> str:
        return "And(" + ", ".join(repr(c) for c in self.children) + ")"


# ---------------------------------------------------------------------------
# Materialised filters
# ---------------------------------------------------------------------------


def _segment_count(mask_long: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Return per-segment sum of ``mask_long`` given per-segment ``lengths``.

    Equivalent to ``[mask_long[off:off+L].sum() for off, L in zip(offsets, lengths)]``
    but vectorised via ``scatter_add_``.
    """
    if lengths.numel() == 0:
        return torch.zeros(0, dtype=lengths.dtype, device=lengths.device)
    seg_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=lengths.device), lengths.to(lengths.device)
    )
    out = torch.zeros(lengths.numel(), dtype=mask_long.dtype, device=mask_long.device)
    out.scatter_add_(0, seg_ids.to(out.device), mask_long)
    return out.to(lengths.dtype)


class MaterializedFilter(ABC):
    """Runtime filter — answers ``is_pid_allowed`` plus a batched ``filter_packed``."""

    def __init__(self, num_passages: int):
        self._num_passages = int(num_passages)

    @property
    def num_passages(self) -> int:
        return self._num_passages

    @property
    @abstractmethod
    def num_allowed(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def contains(self, pids: torch.Tensor) -> torch.Tensor:
        """Boolean membership: returns ``mask`` such that ``mask[i] == True`` iff
        ``pids[i]`` passes the filter. Shape preserved."""

    def filter_pids(self, pids: torch.Tensor) -> torch.Tensor:
        """Filter a 1-D PID tensor. Useful as a drop-in for the legacy
        ``filter_fn`` callback used in :class:`IndexScorer.rank`."""
        if pids.numel() == 0:
            return pids
        mask = self.contains(pids)
        return pids[mask]

    def filter_packed(
        self, packed_pids: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorised in-line filtering pass.

        Args:
          packed_pids: 1-D int tensor [total_len]. Concatenation of per-cell
                       (sorted-ascending) PID lists.
          lengths   : 1-D long tensor [num_cells]. Length of each per-cell list.

        Returns:
          ``(filtered_packed, filtered_lengths)`` with the same packed/lengths
          semantics — each per-cell slice is now restricted to PIDs that pass
          the filter, preserving order.

        The "in-line" property holds because we never materialise the deduped
        union of the unfiltered cell lists; the dedup downstream operates on
        the already-filtered packed tensor.
        """
        assert packed_pids.dim() == 1
        assert lengths.dim() == 1
        if packed_pids.numel() == 0:
            return packed_pids, lengths
        mask = self.contains(packed_pids)
        filtered_packed = packed_pids[mask]
        filtered_lengths = _segment_count(mask.long(), lengths)
        return filtered_packed, filtered_lengths

    def filter_per_cell_reference(
        self, packed_pids: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-cell merge-join reference path.

        Walks each cell list element-by-element and emits only filter-passing
        PIDs. This is the canonical "traversed while doing hit collection"
        idiom — slow in Python, but it's the textbook algorithm and the
        reference layout for a future per-block CUDA kernel (one cell per
        thread block, warp-cooperative merge inside).

        Output is identical to :meth:`filter_packed`.
        """
        offsets = torch.cumsum(lengths, dim=0) - lengths
        out_parts: List[torch.Tensor] = []
        out_lengths: List[int] = []
        for i in range(lengths.numel()):
            off = int(offsets[i].item())
            L = int(lengths[i].item())
            if L == 0:
                out_lengths.append(0)
                continue
            cell = packed_pids[off : off + L]
            m = self.contains(cell)
            kept = cell[m]
            out_parts.append(kept)
            out_lengths.append(int(kept.numel()))
        if out_parts:
            filtered_packed = torch.cat(out_parts)
        else:
            filtered_packed = packed_pids.new_empty(0)
        filtered_lengths = torch.tensor(out_lengths, dtype=lengths.dtype, device=lengths.device)
        return filtered_packed, filtered_lengths


class BitmapFilter(MaterializedFilter):
    """Boolean mask of length ``num_passages``.

    Membership is O(1), branch-free, and trivially coalesced on GPU. Memory is
    ``num_passages`` bits (here implemented as a ``torch.bool`` tensor so it's
    one byte per passage; a packed-bit version is a straightforward swap if
    needed). Always preferred when the index lives on GPU.
    """

    def __init__(self, bitmap: torch.Tensor, num_passages: int):
        super().__init__(num_passages)
        assert bitmap.dtype == torch.bool
        assert bitmap.numel() == num_passages
        self._bitmap = bitmap

    @classmethod
    def from_sorted_pids(cls, sorted_pids: torch.Tensor, num_passages: int) -> "BitmapFilter":
        bitmap = torch.zeros(num_passages, dtype=torch.bool, device=sorted_pids.device)
        if sorted_pids.numel() > 0:
            bitmap[sorted_pids.long()] = True
        return cls(bitmap, num_passages)

    @property
    def num_allowed(self) -> int:
        return int(self._bitmap.sum().item())

    @property
    def device(self) -> torch.device:
        return self._bitmap.device

    @property
    def bitmap(self) -> torch.Tensor:
        return self._bitmap

    def contains(self, pids: torch.Tensor) -> torch.Tensor:
        return self._bitmap[pids.long()]


class SortedListFilter(MaterializedFilter):
    """Sorted-ascending allowed-PID tensor; membership via ``searchsorted``.

    Memory is ``O(N_allowed)`` — preferred when the filter is selective enough
    that the bitmap is wasteful. The internal sorted list is also the natural
    input to a per-cell merge-join CUDA kernel.
    """

    def __init__(self, sorted_pids: torch.Tensor, num_passages: int):
        super().__init__(num_passages)
        if sorted_pids.numel() > 0:
            assert sorted_pids.dim() == 1
        self._sorted_pids = sorted_pids.contiguous()

    @property
    def num_allowed(self) -> int:
        return int(self._sorted_pids.numel())

    @property
    def device(self) -> torch.device:
        return self._sorted_pids.device

    @property
    def sorted_pids(self) -> torch.Tensor:
        return self._sorted_pids

    def contains(self, pids: torch.Tensor) -> torch.Tensor:
        if self._sorted_pids.numel() == 0:
            return torch.zeros(pids.numel(), dtype=torch.bool, device=pids.device)
        # searchsorted requires matching dtypes
        haystack = self._sorted_pids
        if haystack.dtype != pids.dtype:
            haystack = haystack.to(pids.dtype)
        idx = torch.searchsorted(haystack, pids)
        idx_clamped = idx.clamp(max=haystack.numel() - 1)
        in_bounds = idx < haystack.numel()
        return in_bounds & (haystack[idx_clamped] == pids)
