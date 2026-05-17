import torch

from colbert.search.strided_tensor import StridedTensor
from .strided_tensor_core import _create_mask, _create_view


class CandidateGeneration:

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def get_cells(self, Q, ncells):
        scores = (self.codec.centroids @ Q.T)
        if ncells == 1:
            cells = scores.argmax(dim=0, keepdim=True).permute(1, 0)
        else:
            cells = scores.topk(ncells, dim=0, sorted=False).indices.permute(1, 0)  # (32, ncells)
        cells = cells.flatten().contiguous()  # (32 * ncells,)
        cells = cells.unique(sorted=False)
        return cells, scores

    def generate_candidate_eids(self, Q, ncells):
        cells, scores = self.get_cells(Q, ncells)

        eids, cell_lengths = self.ivf.lookup(cells)  # eids = (packedlen,)  lengths = (32 * ncells,)
        eids = eids.long()
        if self.use_gpu:
            eids = eids.cuda()
        return eids, scores

    def generate_candidate_pids(self, Q, ncells):
        cells, scores = self.get_cells(Q, ncells)

        pids, cell_lengths = self.ivf.lookup(cells)
        if self.use_gpu:
            pids = pids.cuda()
        return pids, scores

    def generate_candidate_scores(self, Q, eids):
        E = self.lookup_eids(eids)
        if self.use_gpu:
            E = E.cuda()
        return (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T

    def generate_candidates(self, config, Q, materialized_filter=None):
        """Generate candidate PIDs for query ``Q``.

        When ``materialized_filter`` is not ``None`` the filter's allowed-PID
        set is applied *during* the IVF cell-list traversal — the downstream
        sort/unique runs on the already-filtered packed PIDs rather than on
        the unfiltered union. See ``colbert.search.filtered_search``.
        """
        if materialized_filter is not None:
            from colbert.search.filtered_search import generate_filtered_candidates
            return generate_filtered_candidates(self, config, Q, materialized_filter)

        ncells = config.ncells

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0)
        if self.use_gpu:
            Q = Q.cuda().half()
        assert Q.dim() == 2

        pids, centroid_scores = self.generate_candidate_pids(Q, ncells)

        sorter = pids.sort()
        pids = sorter.values

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        if self.use_gpu:
            pids, pids_counts = pids.cuda(), pids_counts.cuda()

        return pids, centroid_scores
