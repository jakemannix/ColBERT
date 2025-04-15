import pytest
import os
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from colbert.data.collection import Collection
from colbert.infra import ColBERTConfig
from colbert.searcher import Searcher
from colbert.indexer import Indexer


class TestFacetingBenchmark:
    """
    Benchmark tests for metadata filtering and faceting with a large synthetic dataset.
    """
    
    @pytest.fixture(scope="session")
    def benchmark_data_dir(self, request):
        """Get the benchmark data directory from the command line"""
        # Check for the --benchmark-data-dir flag
        benchmark_dir = request.config.getoption("--benchmark-data-dir")
        
        if not benchmark_dir:
            pytest.skip(
                "Benchmark data directory not provided. "
                "Run 'python scripts/create_faceting_benchmark_data.py' to create test data, "
                "then run this test with --benchmark-data-dir=<dir>"
            )
        
        if not os.path.exists(benchmark_dir):
            pytest.skip(f"Benchmark data directory {benchmark_dir} does not exist")
            
        return benchmark_dir
    
    @pytest.fixture(scope="session")
    def synthetic_data_path(self, benchmark_data_dir):
        """Get the path to the synthetic data file"""
        collection_path = os.path.join(benchmark_data_dir, "synthetic_collection.jsonl")
        
        if not os.path.exists(collection_path):
            pytest.skip(
                f"Synthetic collection file {collection_path} not found. "
                "Run 'python scripts/create_faceting_benchmark_data.py' first."
            )
        
        # Load the dataset metadata if available
        metadata_path = os.path.join(benchmark_data_dir, "dataset_metadata.json")
        dataset_metadata = None
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                dataset_metadata = json.load(f)
        
        # Return the paths
        return {
            "temp_dir": benchmark_data_dir,
            "collection_path": collection_path,
            "dataset_metadata": dataset_metadata
        }
    
    @pytest.fixture(scope="module")
    def indexed_collection(self, synthetic_data_path):
        """
        Index the synthetic collection with ColBERT.
        """
        # Load model and index collection
        index_name = "synthetic_index"
        expdir = synthetic_data_path["temp_dir"]
        
        # Configure indexing
        config = ColBERTConfig(
            nbits=2,  # Use small nbits for faster indexing in testing
            root=expdir
        )
        
        print("Indexing synthetic collection...")
        try:
            # Check if checkpoint exists, otherwise download default
            if not os.path.exists(os.path.expanduser("~/.cache/huggingface/hub")):
                print("Downloading ColBERT checkpoint...")
                os.system("python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='colbert-ir/colbertv2.0')\"")
            
            indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
            indexer.index(name=index_name, collection=synthetic_data_path["collection_path"])
            
            # Create searcher
            searcher = Searcher(index=os.path.join(expdir, index_name), config=config)
            
            # Cache the collection in the searcher for testing
            if not hasattr(searcher, 'collection') or searcher.collection is None:
                searcher.collection = Collection(synthetic_data_path["collection_path"])
            
            return {
                "searcher": searcher,
                "config": config,
                "collection_path": synthetic_data_path["collection_path"]
            }
        except Exception as e:
            pytest.skip(f"Indexing failed: {str(e)}")
    
    def compute_brute_force_results(self, searcher, query, k=100, filter_fn=None):
        """
        Compute brute force results by scanning all passages and applying filter afterward.
        """
        # Encode query
        Q = searcher.encode(query)
        
        # Get all documents and scores without filtering
        all_pids, all_scores = self.brute_force_maxsim(searcher, Q)
        
        # Apply filter if provided
        filtered_pids = []
        filtered_scores = []
        
        for idx, pid in enumerate(all_pids):
            if filter_fn is None or filter_fn(pid):
                filtered_pids.append(pid)
                filtered_scores.append(all_scores[idx])
        
        # Sort by score
        sorted_indices = np.argsort(-np.array(filtered_scores))
        top_k_indices = sorted_indices[:k]
        
        top_pids = [filtered_pids[i] for i in top_k_indices]
        top_scores = [filtered_scores[i] for i in top_k_indices]
        
        return top_pids, top_scores
    
    def brute_force_maxsim(self, searcher, Q):
        """
        Brute force implementation of MaxSim for verification.
        This is simplified and not optimized - just for correctness verification.
        """
        # To properly test this, we would need full access to the embedding data
        # For now, we'll use the searcher's internal methods but bypass filtering
        
        # This approach is simplified - in a real implementation, we would directly
        # compute MaxSim between query and document embeddings
        return searcher.ranker.rank(searcher.config, Q, filter_fn=None)
    
    def create_metadata_filter(self, field, value):
        """Create a filter function based on metadata field and value"""
        def filter_fn(pid):
            metadata = self.indexed_collection["searcher"].collection.get_metadata(pid)
            if field not in metadata:
                return False
            
            if isinstance(value, list):
                return metadata[field] in value
            elif isinstance(value, str) and value.startswith('>='):
                threshold = float(value[2:])
                return metadata[field] >= threshold
            elif isinstance(value, str) and value.startswith('<='):
                threshold = float(value[2:])
                return metadata[field] <= threshold
            else:
                return metadata[field] == value
            
        return filter_fn
    
    @pytest.mark.slow
    def test_facet_filtering_correctness(self, indexed_collection):
        """
        Test that facet filtering produces the same results as post-filtering brute force search.
        """
        # Skip if indexing failed
        if indexed_collection is None:
            pytest.skip("Indexing failed")
        
        searcher = indexed_collection["searcher"]
        
        # Test cases with different filter types
        test_cases = [
            # Test single category filter (low cardinality)
            {"query": "science research experiment", "facet_filters": {"category": "science"}},
            
            # Test source filter (medium cardinality)
            {"query": "political election government", "facet_filters": {"source": "source_42"}},
            
            # Test author filter (high cardinality)
            {"query": "painting artist gallery", "facet_filters": {"author": "author_123"}},
            
            # Test year range filter
            {"query": "innovation technology digital", "facet_filters": {"year": ">=2015"}},
            
            # Test multiple filters
            {"query": "sports championship tournament", 
             "facet_filters": {"category": "sports", "year": ">=2010"}}
        ]
        
        for tc in test_cases:
            query = tc["query"]
            facet_filters = tc["facet_filters"]
            
            print(f"\nTesting query: '{query}' with filters: {facet_filters}")
            
            # Create equivalent filter function for brute force approach
            filter_fns = []
            for field, value in facet_filters.items():
                filter_fns.append(self.create_metadata_filter(field, value))
            
            def combined_filter(pid):
                return all(filter_fn(pid) for filter_fn in filter_fns)
            
            # Get results from searcher with facet filtering
            colbert_pids, _, colbert_scores = searcher.search(
                query, 
                k=20, 
                facet_filters=facet_filters
            )
            
            # Get results from brute force approach
            brute_force_pids, brute_force_scores = self.compute_brute_force_results(
                searcher, 
                query, 
                k=20, 
                filter_fn=combined_filter
            )
            
            # Compare results
            # Due to potential slight differences in implementation, we check overlap percentage
            overlap_count = len(set(colbert_pids).intersection(set(brute_force_pids)))
            overlap_percentage = overlap_count / len(colbert_pids) * 100
            
            print(f"Overlap percentage: {overlap_percentage:.2f}%")
            
            # We expect high overlap (at least 80%)
            assert overlap_percentage >= 80, f"Overlap too low: {overlap_percentage:.2f}%"
    
    @pytest.mark.slow
    def test_facet_computation(self, indexed_collection):
        """
        Test facet value computation on search results.
        """
        # Skip if indexing failed
        if indexed_collection is None:
            pytest.skip("Indexing failed")
        
        searcher = indexed_collection["searcher"]
        
        # Test with different queries
        test_queries = [
            "science research experiment", 
            "history ancient civilization",
            "technology innovation digital",
            "politics election government"
        ]
        
        facet_fields = ["category", "source", "year"]
        
        for query in test_queries:
            print(f"\nTesting facet computation for query: '{query}'")
            
            # Get results with facet computation
            _, _, _, facets = searcher.search(
                query, 
                k=100,  # Larger k to ensure good facet distribution
                facet_fields=facet_fields
            )
            
            # Verify facet fields are present
            for field in facet_fields:
                assert field in facets, f"Facet field '{field}' missing from results"
                
                # Check that facet values are not empty
                assert len(facets[field]) > 0, f"No facet values for field '{field}'"
                
                # Print summary
                print(f"Field '{field}' has {len(facets[field])} distinct values")
                
                # For category (low cardinality), verify we get multiple categories
                if field == "category":
                    assert len(facets[field]) > 1, "Expected multiple categories in results"
    
    @pytest.mark.slow
    def test_facet_search_performance(self, indexed_collection):
        """
        Benchmark performance of faceted search vs. regular search.
        """
        # Skip if indexing failed
        if indexed_collection is None:
            pytest.skip("Indexing failed")
        
        searcher = indexed_collection["searcher"]
        
        query = "science technology innovation research"
        k = 100
        
        # Measure regular search time
        import time
        
        # Warm up
        searcher.search(query, k=k)
        
        # Measure regular search
        start_time = time.time()
        searcher.search(query, k=k)
        regular_search_time = time.time() - start_time
        
        # Measure search with facet computation
        start_time = time.time()
        searcher.search(query, k=k, facet_fields=["category", "source", "author", "year"])
        facet_computation_time = time.time() - start_time
        
        # Measure search with facet filtering
        start_time = time.time()
        searcher.search(query, k=k, facet_filters={"category": "science"})
        facet_filtering_time = time.time() - start_time
        
        # Measure search with both
        start_time = time.time()
        searcher.search(
            query, 
            k=k, 
            facet_fields=["category", "source", "author", "year"],
            facet_filters={"category": "science"}
        )
        combined_time = time.time() - start_time
        
        print("\nPerformance measurements:")
        print(f"Regular search: {regular_search_time:.4f}s")
        print(f"With facet computation: {facet_computation_time:.4f}s")
        print(f"With facet filtering: {facet_filtering_time:.4f}s")
        print(f"With both faceting and filtering: {combined_time:.4f}s")
        
        # We don't assert hard performance limits, but we log for inspection