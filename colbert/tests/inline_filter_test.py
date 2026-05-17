"""Unit tests for in-line filtered late-interaction candidate generation.

These tests deliberately avoid the full ``Indexer``/``Searcher`` stack so they
can run on CPU without a model or a real index. They exercise:

  - ``FilterIndex.from_field_data`` (inverted-list construction)
  - ``Term`` / ``And`` / ``Or`` expression materialisation
  - ``BitmapFilter`` and ``SortedListFilter`` filter primitives
  - ``filtered_ivf_lookup`` (vectorised path)
  - ``filtered_ivf_lookup_per_cell`` (reference per-cell merge-join path)
  - End-to-end equivalence of in-line vs post-filter candidate generation,
    against a brute-force Python reference.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import pytest
import torch

from colbert.search.filter_index import (
    And,
    BitmapFilter,
    FilterIndex,
    MaterializedFilter,
    Or,
    SortedListFilter,
    Term,
)
from colbert.search.filtered_search import (
    filtered_ivf_lookup,
    filtered_ivf_lookup_per_cell,
)
from colbert.search.strided_tensor import StridedTensor


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _build_ivf(per_centroid_pids: List[List[int]]) -> StridedTensor:
    """Build a (sorted-ascending) IVF StridedTensor from a list of cell lists."""
    sorted_lists = [sorted(set(c)) for c in per_centroid_pids]
    flat: List[int] = []
    lengths: List[int] = []
    for c in sorted_lists:
        flat.extend(c)
        lengths.append(len(c))
    packed = torch.tensor(flat, dtype=torch.int32)
    lens = torch.tensor(lengths, dtype=torch.long)
    return StridedTensor(packed, lens, use_gpu=False)


def _brute_force_filter(
    per_centroid_pids: List[List[int]],
    cell_ids: List[int],
    allowed_pids: Set[int],
) -> Tuple[List[int], List[int]]:
    """Reference: per-cell intersect with ``allowed_pids``."""
    packed: List[int] = []
    lengths: List[int] = []
    for c in cell_ids:
        cell = sorted(set(per_centroid_pids[c]))
        kept = [p for p in cell if p in allowed_pids]
        packed.extend(kept)
        lengths.append(len(kept))
    return packed, lengths


def _dedup_sort(pids: List[int]) -> List[int]:
    return sorted(set(pids))


# ---------------------------------------------------------------------------
# FilterIndex construction
# ---------------------------------------------------------------------------


def test_filter_index_basic_single_value_field():
    """Single-valued field maps each pid -> one term."""
    fi = FilterIndex.from_field_data(
        field_data={"lang": ["en", "fr", "en", "de", "fr"]},
        num_passages=5,
    )
    assert fi.num_passages == 5
    assert fi.num_terms == 3
    assert fi.has_term("lang", "en")
    assert not fi.has_term("lang", "es")
    assert _dedup_sort(fi.term_pids("lang", "en").tolist()) == [0, 2]
    assert _dedup_sort(fi.term_pids("lang", "fr").tolist()) == [1, 4]
    assert _dedup_sort(fi.term_pids("lang", "de").tolist()) == [3]


def test_filter_index_multi_value_field():
    """Multi-valued field: a passage can be in multiple posting lists."""
    fi = FilterIndex.from_field_data(
        field_data={"tag": [["a", "b"], ["b"], ["a", "c"], []]},
        num_passages=4,
    )
    assert fi.num_terms == 3
    assert _dedup_sort(fi.term_pids("tag", "a").tolist()) == [0, 2]
    assert _dedup_sort(fi.term_pids("tag", "b").tolist()) == [0, 1]
    assert _dedup_sort(fi.term_pids("tag", "c").tolist()) == [2]


def test_filter_index_missing_values_skipped():
    fi = FilterIndex.from_field_data(
        field_data={"region": ["us", None, "eu", None, "us"]},
        num_passages=5,
    )
    assert _dedup_sort(fi.term_pids("region", "us").tolist()) == [0, 4]
    assert _dedup_sort(fi.term_pids("region", "eu").tolist()) == [2]


def test_filter_index_unknown_term_returns_empty():
    fi = FilterIndex.from_field_data(
        field_data={"lang": ["en"]},
        num_passages=1,
    )
    assert fi.term_pids("lang", "zz").numel() == 0
    assert fi.term_pids("missing_field", "anything").numel() == 0


# ---------------------------------------------------------------------------
# Expression materialisation
# ---------------------------------------------------------------------------


@pytest.fixture
def small_index():
    # 10 passages, two fields
    return FilterIndex.from_field_data(
        field_data={
            "lang": ["en", "fr", "en", "de", "fr", "en", "en", "de", "fr", "en"],
            "year": [2020, 2021, 2021, 2020, 2022, 2022, 2020, 2021, 2022, 2022],
        },
        num_passages=10,
    )


def test_term_materialize_matches_index_lookup(small_index):
    mat = Term("lang", "en").materialize(small_index, mode="sorted")
    expected = sorted({0, 2, 5, 6, 9})
    assert _dedup_sort(mat.sorted_pids.tolist()) == expected
    assert mat.num_allowed == len(expected)


def test_or_materialize_is_union(small_index):
    expr = Or(Term("lang", "en"), Term("year", 2022))
    mat = expr.materialize(small_index, mode="sorted")
    expected = sorted({0, 2, 5, 6, 9} | {4, 5, 8, 9})
    assert _dedup_sort(mat.sorted_pids.tolist()) == expected


def test_and_materialize_is_intersection(small_index):
    expr = And(Term("lang", "en"), Term("year", 2022))
    mat = expr.materialize(small_index, mode="sorted")
    expected = sorted({0, 2, 5, 6, 9} & {4, 5, 8, 9})
    assert _dedup_sort(mat.sorted_pids.tolist()) == expected


def test_and_with_no_matches(small_index):
    expr = And(Term("lang", "en"), Term("lang", "fr"))
    mat = expr.materialize(small_index, mode="sorted")
    assert mat.num_allowed == 0


def test_complex_expression(small_index):
    # (lang == en AND year == 2020) OR lang == de
    expr = Or(
        And(Term("lang", "en"), Term("year", 2020)),
        Term("lang", "de"),
    )
    mat = expr.materialize(small_index, mode="sorted")
    en_2020 = {0, 6}  # lang_en = {0,2,5,6,9}, year_2020 = {0,3,6}
    de = {3, 7}
    expected = sorted(en_2020 | de)
    assert _dedup_sort(mat.sorted_pids.tolist()) == expected


# ---------------------------------------------------------------------------
# Auto-mode selection
# ---------------------------------------------------------------------------


def test_auto_picks_sorted_for_sparse_filter():
    # 1000-passage index; filter selects 5 passages → 0.5% density → sorted
    fi = FilterIndex.from_field_data(
        field_data={"f": ["x"] * 5 + ["y"] * 995},
        num_passages=1000,
    )
    mat = Term("f", "x").materialize(fi, mode="auto")
    assert isinstance(mat, SortedListFilter)


def test_auto_picks_bitmap_for_dense_filter():
    # 1000-passage index; filter selects 500 → 50% density → bitmap
    fi = FilterIndex.from_field_data(
        field_data={"f": ["a"] * 500 + ["b"] * 500},
        num_passages=1000,
    )
    mat = Term("f", "a").materialize(fi, mode="auto")
    assert isinstance(mat, BitmapFilter)


# ---------------------------------------------------------------------------
# MaterializedFilter primitives
# ---------------------------------------------------------------------------


@pytest.fixture
def made_filters():
    """Same allowed-PID set, materialised both ways."""
    allowed = sorted({2, 3, 5, 7, 11, 13})
    N = 20
    sorted_t = torch.tensor(allowed, dtype=torch.int32)
    return {
        "allowed": allowed,
        "N": N,
        "sorted": SortedListFilter(sorted_t, N),
        "bitmap": BitmapFilter.from_sorted_pids(sorted_t, N),
    }


def test_contains_matches(made_filters):
    pids = torch.tensor([0, 2, 3, 4, 5, 11, 19, 13], dtype=torch.int32)
    expected = torch.tensor([False, True, True, False, True, True, False, True])
    for name in ("sorted", "bitmap"):
        mat = made_filters[name]
        got = mat.contains(pids)
        assert torch.equal(got, expected), name


def test_filter_packed_matches_per_cell(made_filters):
    # 4 cells, each sorted ascending
    cells = [[0, 1, 2, 3, 5], [4, 6, 7, 11], [9, 13, 15], [12, 14, 16, 17, 19]]
    packed = torch.tensor(sum(cells, []), dtype=torch.int32)
    lengths = torch.tensor([len(c) for c in cells], dtype=torch.long)
    for name in ("sorted", "bitmap"):
        mat = made_filters[name]
        v_pack, v_lens = mat.filter_packed(packed, lengths)
        r_pack, r_lens = mat.filter_per_cell_reference(packed, lengths)
        assert torch.equal(v_pack, r_pack), name
        assert torch.equal(v_lens, r_lens), name
        # also compare against pure-python expected
        allowed = set(made_filters["allowed"])
        expected_per_cell = [[p for p in c if p in allowed] for c in cells]
        expected_packed = sum(expected_per_cell, [])
        expected_lens = [len(c) for c in expected_per_cell]
        assert v_pack.tolist() == expected_packed, name
        assert v_lens.tolist() == expected_lens, name


def test_filter_packed_handles_empty_inputs(made_filters):
    mat = made_filters["bitmap"]
    packed = torch.empty(0, dtype=torch.int32)
    lengths = torch.empty(0, dtype=torch.long)
    p, L = mat.filter_packed(packed, lengths)
    assert p.numel() == 0
    assert L.numel() == 0


def test_filter_packed_all_empty_cells(made_filters):
    """Every cell has length 0 — output must preserve the 0-length cells."""
    mat = made_filters["bitmap"]
    packed = torch.empty(0, dtype=torch.int32)
    lengths = torch.tensor([0, 0, 0], dtype=torch.long)
    p, L = mat.filter_packed(packed, lengths)
    assert p.numel() == 0
    assert torch.equal(L, torch.tensor([0, 0, 0], dtype=torch.long))


def test_filter_packed_no_match_keeps_cell_count(made_filters):
    """All PIDs rejected — output keeps the same number of (zero-length) cells."""
    mat = made_filters["bitmap"]
    cells = [[0, 1], [4, 6], [8, 9]]  # none of these are in allowed = {2,3,5,7,11,13}
    packed = torch.tensor(sum(cells, []), dtype=torch.int32)
    lengths = torch.tensor([len(c) for c in cells], dtype=torch.long)
    p, L = mat.filter_packed(packed, lengths)
    assert p.numel() == 0
    assert L.tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# filtered_ivf_lookup end-to-end
# ---------------------------------------------------------------------------


def test_filtered_ivf_lookup_matches_brute_force(small_index):
    per_centroid = [
        [0, 2, 5, 9],   # centroid 0
        [1, 4, 8],      # centroid 1
        [3, 6, 7],      # centroid 2
        [0, 3, 5, 7],   # centroid 3
        [],             # centroid 4 (empty)
        [1, 2, 8, 9],   # centroid 5
    ]
    ivf = _build_ivf(per_centroid)

    # filter: lang == en (pids {0, 2, 5, 6, 9})
    expr = Term("lang", "en")
    cells_to_probe = torch.tensor([0, 2, 3, 4, 5], dtype=torch.long)

    for mode in ("sorted", "bitmap"):
        mat = expr.materialize(small_index, mode=mode)
        v_pack, v_lens = filtered_ivf_lookup(ivf, cells_to_probe, mat)
        r_pack, r_lens = filtered_ivf_lookup_per_cell(ivf, cells_to_probe, mat)

        exp_pack, exp_lens = _brute_force_filter(
            per_centroid, cells_to_probe.tolist(), {0, 2, 5, 6, 9}
        )

        assert v_pack.tolist() == exp_pack, mode
        assert v_lens.tolist() == exp_lens, mode
        assert r_pack.tolist() == exp_pack, mode
        assert r_lens.tolist() == exp_lens, mode


def test_filtered_ivf_lookup_no_cells():
    per_centroid = [[0, 1], [2, 3]]
    ivf = _build_ivf(per_centroid)
    fi = FilterIndex.from_field_data({"f": ["x", "x", "y", "y"]}, num_passages=4)
    mat = Term("f", "x").materialize(fi, mode="sorted")
    p, L = filtered_ivf_lookup(ivf, torch.tensor([], dtype=torch.long), mat)
    assert p.numel() == 0
    assert L.numel() == 0


def test_filtered_ivf_lookup_filter_matches_nothing(small_index):
    per_centroid = [[0, 1, 2], [3, 4, 5]]
    ivf = _build_ivf(per_centroid)
    # filter for an unknown value -> empty allowed set
    mat = Term("lang", "zz").materialize(small_index, mode="sorted")
    p, L = filtered_ivf_lookup(ivf, torch.tensor([0, 1], dtype=torch.long), mat)
    assert p.numel() == 0
    assert L.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# In-line == post-filter equivalence (the core correctness claim)
# ---------------------------------------------------------------------------


def _simulate_postfilter_candidate_gen(
    ivf: StridedTensor, cells: torch.Tensor, allowed: Set[int]
) -> List[int]:
    """Reference: ColBERT's current path -- union all cells, sort+unique, then filter."""
    parts: List[int] = []
    for c in cells.tolist():
        L = int(ivf.lengths[c].item())
        off = int(ivf.offsets[c].item())
        parts.extend(ivf.tensor[off : off + L].tolist())
    deduped = sorted(set(parts))
    return [p for p in deduped if p in allowed]


def _simulate_inline_candidate_gen(
    ivf: StridedTensor, cells: torch.Tensor, mat: MaterializedFilter
) -> List[int]:
    """New path: filter during cell traversal, then sort+unique."""
    packed, _lens = filtered_ivf_lookup(ivf, cells, mat)
    if packed.numel() == 0:
        return []
    sorted_packed, _ = packed.sort()
    deduped, _ = torch.unique_consecutive(sorted_packed, return_counts=True)
    return deduped.tolist()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_inline_equals_postfilter_randomized(seed):
    """Randomised correctness: for arbitrary IVF + filter + cell selection,
    in-line filtered candidate gen must yield the same deduped candidate PID
    set as the existing post-filter path."""
    rng = random.Random(seed)
    N = 200
    num_centroids = 40
    avg_list_len = 20
    # Random IVF: each centroid gets ~avg_list_len random pids (sorted unique)
    per_centroid = []
    for _ in range(num_centroids):
        k = rng.randint(0, avg_list_len * 2)
        c = sorted({rng.randrange(N) for _ in range(k)})
        per_centroid.append(c)
    ivf = _build_ivf(per_centroid)

    # Random filter: each passage tagged with one of {a,b,c}; we filter for {a} OR {b}
    field = [rng.choice(["a", "b", "c"]) for _ in range(N)]
    fi = FilterIndex.from_field_data({"tag": field}, num_passages=N)

    expr = Or(Term("tag", "a"), Term("tag", "b"))
    allowed_set = {p for p, v in enumerate(field) if v in ("a", "b")}

    # Sample a random subset of centroids to probe
    num_probe = rng.randint(1, num_centroids)
    cells = torch.tensor(
        sorted(rng.sample(range(num_centroids), num_probe)), dtype=torch.long
    )

    expected = _simulate_postfilter_candidate_gen(ivf, cells, allowed_set)

    for mode in ("sorted", "bitmap"):
        mat = expr.materialize(fi, mode=mode)
        got = _simulate_inline_candidate_gen(ivf, cells, mat)
        assert got == expected, f"seed={seed} mode={mode}"


# ---------------------------------------------------------------------------
# Wiring through CandidateGeneration
# ---------------------------------------------------------------------------


class _StubCodec:
    """Minimal stand-in for ResidualCodec; ``CandidateGeneration.get_cells``
    only needs ``codec.centroids``."""
    def __init__(self, centroids: torch.Tensor):
        self.centroids = centroids


class _StubConfig:
    def __init__(self, ncells: int):
        self.ncells = ncells


class _StubCandidateGen:
    """Minimal harness exposing exactly the attributes
    :meth:`CandidateGeneration.generate_candidates` needs."""
    use_gpu = False

    def __init__(self, ivf: StridedTensor, centroids: torch.Tensor):
        self.ivf = ivf
        self.codec = _StubCodec(centroids)


def test_generate_candidates_inline_vs_postfilter():
    """End-to-end: ``CandidateGeneration.generate_candidates`` with and without
    ``materialized_filter`` must produce the same deduped candidate PID set
    (modulo filtering)."""
    from colbert.search.candidate_generation import CandidateGeneration

    rng = random.Random(123)
    N = 80
    num_centroids = 12
    per_centroid = [
        sorted({rng.randrange(N) for _ in range(rng.randint(2, 15))})
        for _ in range(num_centroids)
    ]
    ivf = _build_ivf(per_centroid)

    # Random centroid vectors (low-dim for speed) -- exact values don't matter
    # for correctness; we just need ``get_cells`` to pick *some* cells.
    # Q has shape (1, num_query_tokens, dim) like the real ColBERT path; after
    # the .squeeze(0) inside generate_candidates it becomes (num_query_tokens, dim).
    dim = 4
    num_query_tokens = 6
    centroids = torch.randn(num_centroids, dim)
    Q = torch.randn(1, num_query_tokens, dim)

    cg = _StubCandidateGen(ivf, centroids)
    # bind real methods from CandidateGeneration onto the stub
    cg.get_cells = CandidateGeneration.get_cells.__get__(cg, _StubCandidateGen)
    cg.generate_candidate_pids = CandidateGeneration.generate_candidate_pids.__get__(
        cg, _StubCandidateGen
    )
    cg.generate_candidates = CandidateGeneration.generate_candidates.__get__(
        cg, _StubCandidateGen
    )

    cfg = _StubConfig(ncells=4)

    # Path 1: unfiltered baseline
    unfiltered_pids, _ = cg.generate_candidates(cfg, Q)
    unfiltered_set = set(unfiltered_pids.tolist())

    # Path 2: post-filter callback (legacy mechanism)
    field = [rng.choice(["x", "y", "z"]) for _ in range(N)]
    fi = FilterIndex.from_field_data({"f": field}, num_passages=N)
    allowed = {p for p, v in enumerate(field) if v == "x"}
    postfiltered = sorted(unfiltered_set & allowed)

    # Path 3: in-line filtered candidate gen
    for mode in ("sorted", "bitmap"):
        mat = Term("f", "x").materialize(fi, mode=mode)
        inline_pids, _ = cg.generate_candidates(cfg, Q, materialized_filter=mat)
        inline_set = sorted(set(inline_pids.tolist()))
        assert inline_set == postfiltered, mode


def test_generate_candidates_inline_empty_filter_returns_no_pids():
    from colbert.search.candidate_generation import CandidateGeneration

    per_centroid = [[0, 1, 2], [3, 4, 5]]
    ivf = _build_ivf(per_centroid)
    centroids = torch.randn(2, 4)
    Q = torch.randn(1, 3, 4)

    cg = _StubCandidateGen(ivf, centroids)
    cg.get_cells = CandidateGeneration.get_cells.__get__(cg, _StubCandidateGen)
    cg.generate_candidate_pids = CandidateGeneration.generate_candidate_pids.__get__(
        cg, _StubCandidateGen
    )
    cg.generate_candidates = CandidateGeneration.generate_candidates.__get__(
        cg, _StubCandidateGen
    )
    cfg = _StubConfig(ncells=2)

    fi = FilterIndex.from_field_data({"f": ["x"] * 6}, num_passages=6)
    # Filter for a value that doesn't exist -> empty allowed set
    mat = Term("f", "nope").materialize(fi, mode="sorted")
    pids, _ = cg.generate_candidates(cfg, Q, materialized_filter=mat)
    assert pids.numel() == 0
