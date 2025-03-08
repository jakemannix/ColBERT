import pytest
import os
import json
from colbert.data.collection import Collection

def pytest_addoption(parser):
    """Add custom command line options to pytest"""
    parser.addoption(
        "--benchmark-data-dir",
        action="store",
        default="",
        help="Path to the benchmark data directory for faceting tests"
    )

class TestFacetingData:
    """
    Test the benchmark data loading without running the full benchmark.
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
    
    def test_data_loading(self, benchmark_data_dir):
        """Test that the benchmark data can be loaded and has the expected structure"""
        collection_path = os.path.join(benchmark_data_dir, "synthetic_collection.jsonl")
        metadata_path = os.path.join(benchmark_data_dir, "dataset_metadata.json")
        
        # Verify files exist
        assert os.path.exists(collection_path), f"Collection file not found at {collection_path}"
        assert os.path.exists(metadata_path), f"Metadata file not found at {metadata_path}"
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify metadata structure
        assert "num_passages" in metadata, "Metadata should contain num_passages"
        assert "fields" in metadata, "Metadata should contain fields information"
        
        # Verify fields
        fields = metadata["fields"]
        assert "category" in fields, "Metadata should contain category field"
        assert "source" in fields, "Metadata should contain source field"
        assert "author" in fields, "Metadata should contain author field"
        assert "year" in fields, "Metadata should contain year field"
        
        # Load collection
        collection = Collection(path=collection_path)
        
        # Verify collection size
        assert len(collection) == metadata["num_passages"], "Collection size should match metadata"
        
        # Sample some passages and verify their metadata
        for pid in [0, 10, 20]:
            if pid < len(collection):
                # Access passage
                passage = collection[pid]
                assert isinstance(passage, str), f"Passage {pid} should be a string"
                
                # Access metadata
                passage_metadata = collection.get_metadata(pid)
                assert isinstance(passage_metadata, dict), f"Metadata for passage {pid} should be a dict"
                
                # Verify metadata fields
                assert "category" in passage_metadata, f"Passage {pid} should have category"
                assert "source" in passage_metadata, f"Passage {pid} should have source"
                assert "author" in passage_metadata, f"Passage {pid} should have author"
                assert "year" in passage_metadata, f"Passage {pid} should have year"
                
                # Verify field types
                assert isinstance(passage_metadata["category"], str)
                assert isinstance(passage_metadata["source"], str)
                assert isinstance(passage_metadata["author"], str)
                assert isinstance(passage_metadata["year"], int)
        
        print(f"Successfully loaded benchmark dataset with {len(collection)} passages")
        print(f"Sample categories: {[collection.get_metadata(i)['category'] for i in range(5)]}")
        print(f"Sample years: {[collection.get_metadata(i)['year'] for i in range(5)]}")