import pytest
import os
import json
import random
import tempfile
import numpy as np
from tqdm import tqdm

from colbert.data.collection import Collection

class TestFacetingBenchmarkMock:
    """
    Benchmark tests for faceting using a mock searcher to avoid indexing overhead.
    """
    
    @pytest.fixture(scope="module")
    def synthetic_data(self):
        """Create a medium-sized synthetic data collection with metadata"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="colbert_faceting_bench_")
        collection_path = os.path.join(temp_dir, "synthetic_collection.jsonl")
        
        # Generate synthetic passages with metadata
        num_passages = 1000  # Use 1000 for faster testing
        
        # Define metadata fields with different cardinalities
        category_values = ["science", "history", "politics", "art", "technology", 
                         "health", "business", "entertainment", "sports", "education"]
        sources = [f"source_{i}" for i in range(100)]
        authors = [f"author_{i}" for i in range(500)]  # 500 distinct authors
        
        # Simple word lists for generating content
        common_words = ["the", "and", "of", "to", "in", "is", "that", "it", "with", "for"]
        topics = {
            "science": ["research", "experiment", "theory", "discovery", "scientist"],
            "history": ["ancient", "medieval", "century", "war", "civilization"],
            "politics": ["government", "election", "democracy", "parliament", "president"],
            "art": ["painting", "sculpture", "gallery", "exhibition", "artistic"],
            "technology": ["innovation", "digital", "software", "hardware", "device"]
        }
        
        print(f"Generating {num_passages} synthetic passages with metadata...")
        
        with open(collection_path, 'w') as f:
            for pid in tqdm(range(num_passages)):
                # Assign metadata
                category = random.choice(category_values)
                source = random.choice(sources)
                author = random.choice(authors)
                year = random.randint(2000, 2023)
                
                # Generate simple passage text (just a few words for speed)
                passage_length = random.randint(20, 50)
                topic_words = topics.get(category, topics["science"])
                
                word_choices = []
                for i in range(passage_length):
                    if random.random() < 0.7:
                        word_choices.append(random.choice(common_words))
                    else:
                        word_choices.append(random.choice(topic_words))
                
                passage = " ".join(word_choices)
                
                # Write to JSONL file
                record = {
                    "pid": pid,
                    "passage": passage,
                    "metadata": {
                        "category": category,
                        "source": source,
                        "author": author,
                        "year": year
                    }
                }
                
                f.write(json.dumps(record) + '\n')
        
        # Load the collection
        collection = Collection(path=collection_path)
        
        # Create a mock searcher with minimal functionality
        class MockSearcher:
            def __init__(self):
                self.collection = collection
                self.relevant_pids = {}  # Maps query to list of relevant PIDs with scores
                
                # Create some mock query results
                self._create_mock_results()
            
            def _create_mock_results(self):
                """Create mock search results for testing"""
                # Create mock results for different queries
                
                # For science query, favor science documents
                science_pids = []
                science_scores = []
                for pid in range(num_passages):
                    metadata = self.collection.get_metadata(pid)
                    if metadata.get("category") == "science":
                        score = 0.8 + random.random() * 0.2  # Score between 0.8-1.0
                    else:
                        score = random.random() * 0.7  # Score between 0.0-0.7
                    science_pids.append(pid)
                    science_scores.append(score)
                
                # Sort by descending score
                sorted_indices = np.argsort(-np.array(science_scores))
                self.relevant_pids["science"] = {
                    "pids": [science_pids[i] for i in sorted_indices],
                    "scores": [science_scores[i] for i in sorted_indices]
                }
                
                # Similar approach for history query
                history_pids = []
                history_scores = []
                for pid in range(num_passages):
                    metadata = self.collection.get_metadata(pid)
                    if metadata.get("category") == "history":
                        score = 0.8 + random.random() * 0.2
                    else:
                        score = random.random() * 0.7
                    history_pids.append(pid)
                    history_scores.append(score)
                
                sorted_indices = np.argsort(-np.array(history_scores))
                self.relevant_pids["history"] = {
                    "pids": [history_pids[i] for i in sorted_indices],
                    "scores": [history_scores[i] for i in sorted_indices]
                }
                
                # For art query
                art_pids = []
                art_scores = []
                for pid in range(num_passages):
                    metadata = self.collection.get_metadata(pid)
                    if metadata.get("category") == "art":
                        score = 0.8 + random.random() * 0.2
                    else:
                        score = random.random() * 0.7
                    art_pids.append(pid)
                    art_scores.append(score)
                
                sorted_indices = np.argsort(-np.array(art_scores))
                self.relevant_pids["art"] = {
                    "pids": [art_pids[i] for i in sorted_indices],
                    "scores": [art_scores[i] for i in sorted_indices]
                }
            
            def search(self, query, k=10, facet_fields=None, facet_filters=None):
                """Mock search function"""
                # Get pre-computed results for this query
                query_key = query.split()[0].lower()  # Use first word as key
                if query_key not in self.relevant_pids:
                    query_key = "science"  # Default to science
                
                results = self.relevant_pids[query_key]
                all_pids = results["pids"]
                all_scores = results["scores"]
                
                # Apply facet filters if provided
                if facet_filters:
                    filter_fn = self._create_facet_filter(None, facet_filters)
                    filtered_pids = []
                    filtered_scores = []
                    
                    for i, pid in enumerate(all_pids):
                        if filter_fn(pid):
                            filtered_pids.append(pid)
                            filtered_scores.append(all_scores[i])
                    
                    pids = filtered_pids[:k]
                    scores = filtered_scores[:k]
                else:
                    pids = all_pids[:k]
                    scores = all_scores[:k]
                
                # Calculate ranks
                ranks = list(range(1, len(pids) + 1))
                
                # Compute facets if requested
                if facet_fields:
                    facets = self._compute_facets(pids, facet_fields)
                    return pids, ranks, scores, facets
                
                return pids, ranks, scores
            
            def _create_facet_filter(self, original_filter_fn, facet_filters):
                """Create a filter function based on facet filters"""
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
                """Compute facet counts for the given fields"""
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
        
        # Create mock searcher
        searcher = MockSearcher()
        
        return {
            "collection": collection,
            "searcher": searcher,
            "temp_dir": temp_dir,
            "collection_path": collection_path
        }
    
    def test_facet_filtering(self, synthetic_data):
        """Test facet filtering with mock searcher"""
        searcher = synthetic_data["searcher"]
        
        # Test simple category filter
        pids, ranks, scores = searcher.search("science", k=50, facet_filters={"category": "science"})
        
        # Verify all returned documents are in science category
        for pid in pids:
            metadata = searcher.collection.get_metadata(pid)
            assert metadata["category"] == "science"
        
        # Test year range filter
        pids, ranks, scores = searcher.search("history", k=50, facet_filters={"year": ">=2020"})
        
        # Verify all returned documents are from 2020 or later
        for pid in pids:
            metadata = searcher.collection.get_metadata(pid)
            assert metadata["year"] >= 2020
        
        # Test combined filters
        pids, ranks, scores = searcher.search("art", k=50, 
                                           facet_filters={"category": "art", "year": ">=2015"})
        
        # Verify all returned documents meet both criteria
        for pid in pids:
            metadata = searcher.collection.get_metadata(pid)
            assert metadata["category"] == "art"
            assert metadata["year"] >= 2015
    
    def test_facet_computation(self, synthetic_data):
        """Test facet computation with mock searcher"""
        searcher = synthetic_data["searcher"]
        
        # Test facet computation on science query
        _, _, _, facets = searcher.search("science", k=100, 
                                     facet_fields=["category", "source", "year"])
        
        # Verify facet structure
        assert "category" in facets
        assert "source" in facets
        assert "year" in facets
        
        # Verify we have values in each facet
        assert len(facets["category"]) >= 1
        assert len(facets["source"]) >= 1
        assert len(facets["year"]) >= 1
        
        # Examine distribution of values
        print("\nScience query facets:")
        print(f"Categories: {len(facets['category'])} unique values")
        print(f"Sources: {len(facets['source'])} unique values")
        print(f"Years: {len(facets['year'])} unique values")
        
        # Test facet computation with filtering
        _, _, _, facets = searcher.search("history", k=100, 
                                     facet_fields=["category", "source", "year"],
                                     facet_filters={"category": "history"})
        
        # Verify that all documents are from history category
        assert len(facets["category"]) == 1
        assert "history" in facets["category"]
        
        # We should still have multiple sources and years
        assert len(facets["source"]) > 1
        assert len(facets["year"]) > 1
        
        print("\nHistory query facets (filtered to history category):")
        print(f"Categories: {len(facets['category'])} unique values")
        print(f"Sources: {len(facets['source'])} unique values")
        print(f"Years: {len(facets['year'])} unique values")
    
    def test_facet_performance(self, synthetic_data):
        """Test performance of facet operations"""
        import time
        searcher = synthetic_data["searcher"]
        
        # Warm up
        searcher.search("science", k=100)
        
        # Test regular search
        start_time = time.time()
        searcher.search("science", k=100)
        regular_time = time.time() - start_time
        
        # Test with facet computation
        start_time = time.time()
        searcher.search("science", k=100, facet_fields=["category", "source", "author", "year"])
        facet_time = time.time() - start_time
        
        # Test with facet filtering
        start_time = time.time()
        searcher.search("science", k=100, facet_filters={"category": "science"})
        filter_time = time.time() - start_time
        
        # Test with both
        start_time = time.time()
        searcher.search("science", k=100, 
                    facet_fields=["category", "source", "author", "year"],
                    facet_filters={"category": "science"})
        combined_time = time.time() - start_time
        
        # Print performance results
        print("\nPerformance measurements (mock searcher):")
        print(f"Regular search: {regular_time:.6f}s")
        print(f"With facet computation: {facet_time:.6f}s")
        print(f"With facet filtering: {filter_time:.6f}s")
        print(f"With both faceting and filtering: {combined_time:.6f}s")
        
        # Facet computation should add some overhead
        assert facet_time > regular_time
        
        # Calculate overhead percentages
        facet_overhead = (facet_time - regular_time) / regular_time * 100
        filter_overhead = (filter_time - regular_time) / regular_time * 100
        combined_overhead = (combined_time - regular_time) / regular_time * 100
        
        print(f"Facet computation overhead: {facet_overhead:.1f}%")
        print(f"Filtering overhead: {filter_overhead:.1f}%")
        print(f"Combined overhead: {combined_overhead:.1f}%")