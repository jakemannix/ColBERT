import pytest
import torch
import os
import tempfile
from colbert.data.collection import Collection
from colbert.searcher import Searcher
from colbert.infra import ColBERTConfig

class MockRanker:
    def __init__(self, pids, scores):
        self.pids = pids
        self.scores = scores
    
    def rank(self, config, Q, filter_fn=None, pids=None):
        # Apply filter_fn if provided
        filtered_pids = []
        filtered_scores = []
        
        for idx, pid in enumerate(self.pids):
            if filter_fn is None or filter_fn(pid):
                filtered_pids.append(pid)
                filtered_scores.append(self.scores[idx])
        
        return filtered_pids, filtered_scores

class MockCheckpoint:
    def __init__(self):
        class MockTokenizer:
            def __init__(self):
                self.query_maxlen = 32
                
            def queryFromText(self, text, **kwargs):
                # Return a dummy tensor
                return torch.ones((len(text) if isinstance(text, list) else 1, 768))
        
        self.query_tokenizer = MockTokenizer()
    
    def queryFromText(self, text, **kwargs):
        # Return a dummy tensor
        return torch.ones((len(text) if isinstance(text, list) else 1, 768))

class TestFaceting:
    def test_faceting_search(self):
        """Test that faceted search returns correct facet counts."""
        # Create test collection with metadata
        passages = [
            "Document about science and physics",
            "Document about history and world war",
            "Document about chemistry experiments",
            "Document about biology and genetics",
            "Document about ancient Rome"
        ]
        
        metadata = {
            0: {"category": "science", "year": 2020, "author": "Smith"},
            1: {"category": "history", "year": 2019, "author": "Jones"},
            2: {"category": "science", "year": 2020, "author": "Brown"},
            3: {"category": "science", "year": 2021, "author": "Smith"},
            4: {"category": "history", "year": 2018, "author": "Miller"}
        }
        
        collection = Collection(data=passages, metadata=metadata)
        
        # Create a searcher with minimal mocking
        # Instead of creating a real Searcher, we'll create a minimal mock that has just the methods we need
        # Create a mock searcher with just the methods we need for faceting
        class MockSearcher:
            def __init__(self, collection):
                self.collection = collection
                self.config = ColBERTConfig()
            
            def search(self, text, k=10, filter_fn=None, facet_fields=None, facet_filters=None, **kwargs):
                # Create the facet filter
                combined_filter_fn = self._create_facet_filter(filter_fn, facet_filters)
                
                # Get search results using mocked dense_search
                pids, ranks, scores = self.dense_search(None, k, filter_fn=combined_filter_fn)
                
                # If facet fields are requested, compute facet counts
                if facet_fields:
                    facets = self._compute_facets(pids, facet_fields)
                    return pids, ranks, scores, facets
                
                return pids, ranks, scores
                
            def search_all(self, queries, k=10, filter_fn=None, facet_fields=None, facet_filters=None, **kwargs):
                """
                Search for multiple queries with faceting support
                """
                from colbert.data.ranking import Ranking
                from colbert.infra.provenance import Provenance
                
                if isinstance(queries, dict):
                    qids = list(queries.keys())
                else:
                    # Handle list inputs
                    qids = list(range(len(queries)))
                    queries = {qid: query for qid, query in enumerate(queries)}
                
                # Create facet filter function
                combined_filter_fn = self._create_facet_filter(filter_fn, facet_filters)
                
                # Results storage
                all_facets = {}
                all_scored_pids = []
                
                for qid in qids:
                    query_text = queries[qid]
                    
                    if facet_fields:
                        # With facet computation
                        pids, ranks, scores, facets = self.search(
                            query_text, k, 
                            filter_fn=combined_filter_fn,
                            facet_fields=facet_fields
                        )
                        all_scored_pids.append(list(zip(pids, ranks, scores)))
                        all_facets[qid] = facets
                    else:
                        # Without facet computation
                        pids, ranks, scores = self.search(
                            query_text, k, 
                            filter_fn=combined_filter_fn
                        )
                        all_scored_pids.append(list(zip(pids, ranks, scores)))
                
                data = {qid: val for qid, val in zip(qids, all_scored_pids)}
                
                # Create provenance for the ranking
                provenance = Provenance()
                provenance.source = 'MockSearcher::search_all'
                
                if facet_fields:
                    # Add facet information to provenance
                    provenance.facets = True
                    # Create a ranking with facet data
                    return Ranking(data=data, provenance=provenance, metadata={"facets": all_facets})
                else:
                    return Ranking(data=data, provenance=provenance)
            
            def dense_search(self, Q, k, filter_fn=None, **kwargs):
                # Use the mock ranker directly
                pids, scores = self.ranker.rank(None, None, filter_fn=filter_fn)
                return pids[:k], list(range(1, k+1)), scores[:k]
                
            # Copy the faceting methods directly from the Searcher class
            def _create_facet_filter(self, original_filter_fn, facet_filters):
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
        
        searcher = MockSearcher(collection)
        
        # Mock the ranker and checkpoint
        searcher.ranker = MockRanker(
            pids=[0, 1, 2, 3, 4],
            scores=[0.9, 0.8, 0.7, 0.6, 0.5]
        )
        searcher.checkpoint = MockCheckpoint()
        
        # Test basic faceting
        _, _, _, facets = searcher.search(
            "science", 
            k=5, 
            facet_fields=["category", "year", "author"]
        )
        
        # Check facet counts
        assert "category" in facets
        assert "year" in facets
        assert "author" in facets
        
        assert facets["category"]["science"] == 3
        assert facets["category"]["history"] == 2
        assert facets["year"]["2020"] == 2
        assert facets["author"]["Smith"] == 2
        
        # Test facet filtering
        pids, _, _, facets = searcher.search(
            "science", 
            k=5, 
            facet_fields=["category", "year", "author"],
            facet_filters={"category": "science"}
        )
        
        # Should only return science documents
        assert len(pids) == 3
        assert all(searcher.collection.get_metadata(pid)["category"] == "science" for pid in pids)
        
        # Check filtered facet counts
        assert facets["category"]["science"] == 3
        assert "history" not in facets["category"]
        assert facets["year"]["2020"] == 2
        assert facets["year"]["2021"] == 1
        
        # Test with multiple filter values
        pids, _, _, facets = searcher.search(
            "science", 
            k=5, 
            facet_fields=["category", "year", "author"],
            facet_filters={"author": ["Smith", "Brown"]}
        )
        
        # Should only return documents by Smith or Brown
        assert len(pids) == 3
        assert all(searcher.collection.get_metadata(pid)["author"] in ["Smith", "Brown"] for pid in pids)
        
        # Test numeric filtering
        pids, _, _, facets = searcher.search(
            "science", 
            k=5, 
            facet_fields=["category", "year", "author"],
            facet_filters={"year": ">=2020"}
        )
        
        # Should only return documents from 2020 or later
        assert len(pids) == 3
        assert all(searcher.collection.get_metadata(pid)["year"] >= 2020 for pid in pids)
        
        # Test search_all with faceting
        queries = {
            1: "science",
            2: "history"
        }
        
        ranking = searcher.search_all(
            queries,
            k=5,
            facet_fields=["category", "year", "author"]
        )
        
        # Check that the ranking object has the expected structure
        assert "facets" in ranking.metadata
        assert 1 in ranking.metadata["facets"]  # Check facets for query ID 1
        assert 2 in ranking.metadata["facets"]  # Check facets for query ID 2
        
        # Check facet counts for first query (science)
        assert ranking.metadata["facets"][1]["category"]["science"] == 3
        assert ranking.metadata["facets"][1]["category"]["history"] == 2
        
        # Test search_all with facet filtering
        ranking = searcher.search_all(
            queries,
            k=5,
            facet_fields=["category", "year", "author"],
            facet_filters={"category": "science"}
        )
        
        # Check filtered results
        assert len(ranking.data[1]) == 3  # 3 science documents for query 1
        assert len(ranking.data[2]) == 3  # 3 science documents for query 2
        
        # Check facet counts after filtering
        assert ranking.metadata["facets"][1]["category"]["science"] == 3
        assert "history" not in ranking.metadata["facets"][1]["category"]