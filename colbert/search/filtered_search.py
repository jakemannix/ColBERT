"""
Filtered candidate generation: walk the IVF cell lists with a filter applied
*during* the traversal, never materialising the deduped union of the
unfiltered cell lists.

Two equivalent code paths are provided:

  - :func:`filtered_ivf_lookup`            : vectorised, batched over all cells.
                                              Single pass over packed PIDs.
                                              Easy to lower to a fused CUDA
                                              kernel.
  - :func:`filtered_ivf_lookup_per_cell`   : per-cell merge-join reference
                                              implementation. Slow in Python
                                              but exists as the canonical
                                              algorithm and as the layout
                                              template for a per-block kernel.

Both consume an ``ivf`` :class:`StridedTensor` (centroid -> sorted PID list)
and a :class:`MaterializedFilter`.
"""

from __future__ import annotations

from typing import Tuple

import torch

from colbert.search.filter_index import MaterializedFilter
from colbert.search.strided_tensor import StridedTensor


def _gather_packed_cells(
    ivf: StridedTensor, cell_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice the underlying packed IVF tensor for the given cells.

    Returns ``(packed_pids, lengths)`` with one segment per ``cell_ids[i]``,
    in input order. The returned ``packed_pids`` matches what
    :meth:`StridedTensor.lookup` would produce in ``'packed'`` mode but without
    going through the C++ segmented_lookup extension — that lets the filtered
    path stay pure-torch and load-extension-free for unit tests.
    """
    cell_ids = cell_ids.long().cpu()
    lengths = ivf.lengths[cell_ids]
    offsets = ivf.offsets[cell_ids]
    if cell_ids.numel() == 0:
        return ivf.tensor.new_empty(0), lengths
    starts = offsets.tolist()
    lens = lengths.tolist()
    parts = [ivf.tensor[s : s + L] for s, L in zip(starts, lens) if L > 0]
    if parts:
        packed = torch.cat(parts)
    else:
        packed = ivf.tensor.new_empty(0)
    return packed, lengths


def filtered_ivf_lookup(
    ivf: StridedTensor,
    cell_ids: torch.Tensor,
    materialized_filter: MaterializedFilter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised filtered IVF lookup.

    Walks the (packed) per-cell PID lists once, applies the filter pointwise,
    and returns the compacted output along with per-cell filtered lengths.

    Args:
      ivf                  : :class:`StridedTensor` indexed by centroid id.
      cell_ids             : 1-D int tensor of centroid ids to retrieve.
      materialized_filter  : a :class:`MaterializedFilter`.

    Returns:
      ``(filtered_packed_pids, filtered_lengths)``.
    """
    packed_pids, lengths = _gather_packed_cells(ivf, cell_ids)
    target_device = materialized_filter.device
    if packed_pids.device != target_device:
        packed_pids = packed_pids.to(target_device)
    if lengths.device != target_device:
        lengths = lengths.to(target_device)
    return materialized_filter.filter_packed(packed_pids, lengths)


def filtered_ivf_lookup_per_cell(
    ivf: StridedTensor,
    cell_ids: torch.Tensor,
    materialized_filter: MaterializedFilter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-cell merge-join reference implementation.

    Functionally identical to :func:`filtered_ivf_lookup` but loops over
    individual cells. Use only as a correctness/teaching reference; for the
    real CPU path use the vectorised form. For the GPU full-VRAM path, a
    future CUDA kernel can mirror this loop with one block per cell.
    """
    packed_pids, lengths = _gather_packed_cells(ivf, cell_ids)
    target_device = materialized_filter.device
    if packed_pids.device != target_device:
        packed_pids = packed_pids.to(target_device)
    if lengths.device != target_device:
        lengths = lengths.to(target_device)
    return materialized_filter.filter_per_cell_reference(packed_pids, lengths)


def generate_filtered_candidate_pids(
    candidate_generator,
    Q: torch.Tensor,
    ncells: int,
    materialized_filter: MaterializedFilter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equivalent of :meth:`CandidateGeneration.generate_candidate_pids` but
    with in-line filtering applied during the IVF traversal.

    Returns the (still-duplicated, still-unsorted-across-cells)
    filtered packed PIDs plus the centroid score matrix. The downstream
    dedup/sort step in :meth:`CandidateGeneration.generate_candidates` will
    operate on this filtered output, so it never sees the unfiltered union.
    """
    cells, scores = candidate_generator.get_cells(Q, ncells)
    pids, _filtered_lengths = filtered_ivf_lookup(
        candidate_generator.ivf, cells, materialized_filter
    )
    if candidate_generator.use_gpu:
        pids = pids.cuda()
    return pids, scores


def generate_filtered_candidates(
    candidate_generator,
    config,
    Q: torch.Tensor,
    materialized_filter: MaterializedFilter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for :meth:`CandidateGeneration.generate_candidates`.

    Same shape of return values, but the filter is applied while collecting
    hits from the cluster (centroid) lists rather than as a post-step.
    """
    ncells = config.ncells

    assert isinstance(candidate_generator.ivf, StridedTensor)

    Q = Q.squeeze(0)
    if candidate_generator.use_gpu:
        Q = Q.cuda().half()
    assert Q.dim() == 2

    pids, centroid_scores = generate_filtered_candidate_pids(
        candidate_generator, Q, ncells, materialized_filter
    )

    if pids.numel() == 0:
        empty = pids
        counts = empty.new_empty(0, dtype=torch.long)
        if candidate_generator.use_gpu:
            empty, counts = empty.cuda(), counts.cuda()
        return empty, centroid_scores

    sorter = pids.sort()
    pids = sorter.values
    pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
    if candidate_generator.use_gpu:
        pids, pids_counts = pids.cuda(), pids_counts.cuda()
    return pids, centroid_scores
