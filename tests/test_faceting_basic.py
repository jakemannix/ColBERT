import pytest
import os
import json
import tempfile
from colbert.data.collection import Collection
from colbert.searcher import Searcher

class TestFacetingBasic:
    """
    Basic tests for faceting functionality without the overhead of indexing.
    """
    
    @pytest.fixture
    def mock_collection(self):
        """Create a small collection with metadata for basic testing"""
        # Create temporary file
        temp_dir = tempfile.mkdtemp(prefix="colbert_faceting_basic_")
        collection_path = os.path.join(temp_dir, "mini_collection.jsonl")
        
        # Create a small test collection with 10 documents
        documents = [
            {"pid": 0, "passage": "Document about science and physics", 
             "metadata": {"category": "science", "year": 2020, "author": "Smith"}},
            {"pid": 1, "passage": "Document about history and world war", 
             "metadata": {"category": "history", "year": 2019, "author": "Jones"}},
            {"pid": 2, "passage": "Document about chemistry experiments", 
             "metadata": {"category": "science", "year": 2020, "author": "Brown"}},
            {"pid": 3, "passage": "Document about biology and genetics", 
             "metadata": {"category": "science", "year": 2021, "author": "Smith"}},
            {"pid": 4, "passage": "Document about ancient Rome", 
             "metadata": {"category": "history", "year": 2018, "author": "Miller"}},
            {"pid": 5, "passage": "Document about quantum physics", 
             "metadata": {"category": "science", "year": 2022, "author": "Johnson"}},
            {"pid": 6, "passage": "Document about World War II", 
             "metadata": {"category": "history", "year": 2017, "author": "Williams"}},
            {"pid": 7, "passage": "Document about cell biology", 
             "metadata": {"category": "science", "year": 2021, "author": "Davis"}},
            {"pid": 8, "passage": "Document about Renaissance art", 
             "metadata": {"category": "art", "year": 2019, "author": "Wilson"}},
            {"pid": 9, "passage": "Document about modern technology", 
             "metadata": {"category": "technology", "year": 2023, "author": "Moore"}}
        ]
        
        # Write to JSONL file
        with open(collection_path, 'w') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        
        # Create collection object
        collection = Collection(path=collection_path)
        
        return {
            "collection": collection,
            "temp_dir": temp_dir,
            "collection_path": collection_path
        }
    
    def test_collection_metadata_loading(self, mock_collection):
        """Test that metadata is properly loaded from JSONL"""
        collection = mock_collection["collection"]
        
        # Test metadata retrieval
        assert collection.get_metadata(0)["category"] == "science"
        assert collection.get_metadata(1)["year"] == 2019
        assert collection.get_metadata(3)["author"] == "Smith"
        assert collection.get_metadata(9)["category"] == "technology"
        
        # Test non-existent metadata
        assert collection.get_metadata(100) == {}
    
    def test_metadata_filter_function(self, mock_collection):
        """Test that the metadata filter function works correctly"""
        collection = mock_collection["collection"]
        
        # Create a simple searcher mock just for testing the filter function
        class MockSearcher:
            def __init__(self, collection):
                self.collection = collection
                
            def _create_facet_filter(self, original_filter_fn, facet_filters):
                """Copy of the method from searcher.py"""
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
        
        searcher = MockSearcher(collection)
        
        # Test simple category filter
        category_filter = searcher._create_facet_filter(None, {"category": "science"})
        science_pids = [pid for pid in range(10) if category_filter(pid)]
        assert science_pids == [0, 2, 3, 5, 7]
        
        # Test year range filter
        year_filter = searcher._create_facet_filter(None, {"year": ">=2020"})
        recent_pids = [pid for pid in range(10) if year_filter(pid)]
        assert recent_pids == [0, 2, 3, 5, 7, 9]
        
        # Test multiple filters
        combined_filter = searcher._create_facet_filter(None, {
            "category": "science", 
            "year": ">=2021"
        })
        filtered_pids = [pid for pid in range(10) if combined_filter(pid)]
        assert filtered_pids == [3, 5, 7]
        
        # Test filter with list of values
        multi_filter = searcher._create_facet_filter(None, {
            "category": ["science", "art"]
        })
        filtered_pids = [pid for pid in range(10) if multi_filter(pid)]
        assert filtered_pids == [0, 2, 3, 5, 7, 8]
    
    def test_facet_computation(self, mock_collection):
        """Test facet value computation"""
        collection = mock_collection["collection"]
        
        # Create a simple function to compute facets (derived from searcher.py)
        def compute_facets(pids, facet_fields):
            facets = {}
            
            for field in facet_fields:
                facets[field] = {}
                
                for pid in pids:
                    metadata = collection.get_metadata(pid)
                    
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
        
        # Test with all documents
        all_pids = list(range(10))
        facets = compute_facets(all_pids, ["category", "year", "author"])
        
        # Check category facets
        assert facets["category"]["science"] == 5
        assert facets["category"]["history"] == 3
        assert facets["category"]["art"] == 1
        assert facets["category"]["technology"] == 1
        
        # Check year facets
        assert facets["year"]["2020"] == 2
        assert facets["year"]["2021"] == 2
        assert facets["year"]["2019"] == 2
        
        # Test with filtered documents (only science category)
        science_pids = [0, 2, 3, 5, 7]
        facets = compute_facets(science_pids, ["year", "author"])
        
        # Check year facets for science documents
        assert facets["year"]["2020"] == 2
        assert facets["year"]["2021"] == 2
        assert facets["year"]["2022"] == 1
        
        # Check author facets for science documents
        assert facets["author"]["Smith"] == 2
        assert facets["author"]["Brown"] == 1
        assert facets["author"]["Johnson"] == 1
        assert facets["author"]["Davis"] == 1