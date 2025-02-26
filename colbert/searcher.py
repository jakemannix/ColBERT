import os
import torch

from tqdm import tqdm
from typing import Union

from colbert.data import Collection, Queries, Ranking

from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats

import time

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None, index_root=None, verbose:int = 3):
        self.verbose = verbose
        if self.verbose > 1:
            print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config, verbose=self.verbose)
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and use_gpu:
            raise ValueError(f"Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, use_gpu, load_index_with_mmap)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries, full_length_search=False):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)

        return Q

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False, pids=None, 
              facet_fields=None, facet_filters=None):
        """
        Search for passages matching the given query.
        
        Args:
            text: The search query text
            k: Number of results to return
            filter_fn: Function to filter results (used by base implementation)
            full_length_search: Whether to use full-length search
            pids: List of passage IDs to restrict search to
            facet_fields: List of metadata fields to compute facet counts for
            facet_filters: Dict of metadata fields to filter values, e.g. {"year": [2019, 2020], "category": "science"}
            
        Returns:
            A tuple of (pids, ranks, scores) if facet_fields is None, otherwise
            a tuple of (pids, ranks, scores, facets) where facets is a dict of field -> value -> count
        """
        Q = self.encode(text, full_length_search=full_length_search)
        
        # Create a filter function that combines the original filter with facet filtering
        combined_filter_fn = self._create_facet_filter(filter_fn, facet_filters)
        
        # Get search results
        pids, ranks, scores = self.dense_search(Q, k, filter_fn=combined_filter_fn, pids=pids)
        
        # If facet fields are requested, compute facet counts
        if facet_fields:
            facets = self._compute_facets(pids, facet_fields)
            return pids, ranks, scores, facets
        
        return pids, ranks, scores

    def search_all(self, queries: TextQueries, k=10, filter_fn=None, full_length_search=False, 
                 qid_to_pids=None, facet_fields=None, facet_filters=None):
        """
        Search for multiple queries with optional faceting.
        
        Args:
            queries: The search queries
            k: Number of results to return per query
            filter_fn: Function to filter results
            full_length_search: Whether to use full-length search
            qid_to_pids: Dict of query ID to list of passage IDs to restrict search to
            facet_fields: List of metadata fields to compute facet counts for
            facet_filters: Dict of metadata fields to filter values
            
        Returns:
            A Ranking object, optionally with facet information
        """
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn, qid_to_pids=qid_to_pids,
                                facet_fields=facet_fields, facet_filters=facet_filters)

    def _search_all_Q(self, queries, Q, k, filter_fn=None, qid_to_pids=None, 
                   facet_fields=None, facet_filters=None):
        qids = list(queries.keys())

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        # Create the facet filter function once
        combined_filter_fn = self._create_facet_filter(filter_fn, facet_filters)
        
        # Track facets separately if requested
        all_facets = {}
        all_scored_pids = []
        
        for query_idx, qid in tqdm(enumerate(qids)):
            # Use the combined filter function for search
            if facet_fields:
                # With facet computation
                pids, ranks, scores, facets = self.search(
                    queries[qid], k, 
                    filter_fn=combined_filter_fn,
                    pids=qid_to_pids[qid],
                    facet_fields=facet_fields
                )
                all_scored_pids.append(list(zip(pids, ranks, scores)))
                all_facets[qid] = facets
            else:
                # Without facet computation
                search_results = self.dense_search(
                    Q[query_idx:query_idx+1],
                    k, filter_fn=combined_filter_fn,
                    pids=qid_to_pids[qid]
                )
                all_scored_pids.append(list(zip(*search_results)))

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k
        
        if facet_fields:
            # Add facet information to provenance
            provenance.facets = True
            # Create a ranking with facet data
            return Ranking(data=data, provenance=provenance, metadata={"facets": all_facets})
        else:
            return Ranking(data=data, provenance=provenance)

    def _create_facet_filter(self, original_filter_fn, facet_filters):
        """
        Create a filter function that combines the original filter with facet filtering.
        
        Args:
            original_filter_fn: The original filter function
            facet_filters: Dict of metadata fields to filter values
            
        Returns:
            A filter function that combines both filters
        """
        if not facet_filters:
            return original_filter_fn
            
        def combined_filter(pid):
            # Apply original filter if it exists
            if original_filter_fn and not original_filter_fn(pid):
                return False
                
            # Apply facet filters
            metadata = self.collection.get_metadata(pid)
            
            for field, filter_values in facet_filters.items():
                if field not in metadata:
                    return False
                    
                field_value = metadata[field]
                
                # Handle different filter types
                if isinstance(filter_values, list):
                    # List of accepted values
                    if field_value not in filter_values:
                        return False
                elif isinstance(filter_values, str) and filter_values.startswith('>='):
                    # Numeric greater-than-or-equal filter
                    threshold = float(filter_values[2:])
                    if not isinstance(field_value, (int, float)) or field_value < threshold:
                        return False
                elif isinstance(filter_values, str) and filter_values.startswith('<='):
                    # Numeric less-than-or-equal filter
                    threshold = float(filter_values[2:])
                    if not isinstance(field_value, (int, float)) or field_value > threshold:
                        return False
                elif filter_values != field_value:
                    # Exact match
                    return False
                    
            return True
            
        return combined_filter
        
    def _compute_facets(self, pids, facet_fields):
        """
        Compute facet counts for the given fields.
        
        Args:
            pids: List of passage IDs
            facet_fields: List of metadata fields to compute facets for
            
        Returns:
            Dict of field -> value -> count
        """
        facets = {}
        
        for field in facet_fields:
            facets[field] = {}
            
            for pid in pids:
                metadata = self.collection.get_metadata(pid)
                
                if field in metadata:
                    value = metadata[field]
                    
                    # Convert value to string for consistent keys
                    if isinstance(value, (list, tuple)):
                        # Handle multi-valued fields
                        for v in value:
                            v_str = str(v)
                            facets[field][v_str] = facets[field].get(v_str, 0) + 1
                    else:
                        value_str = str(value)
                        facets[field][value_str] = facets[field].get(value_str, 0) + 1
        
        return facets

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None, pids=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids)

        return pids[:k], list(range(1, k+1)), scores[:k]
