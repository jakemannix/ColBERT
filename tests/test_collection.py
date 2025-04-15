import os
import tempfile
import pytest
from colbert.data.collection import Collection

class TestCollection:
    def test_create_empty_collection(self):
        """Test creating an empty collection."""
        collection = Collection()
        assert len(collection) == 0
        
    def test_create_collection_from_data(self):
        """Test creating a collection from a list of passages."""
        passages = ["This is passage 1", "This is passage 2", "This is passage 3"]
        collection = Collection(data=passages)
        
        assert len(collection) == 3
        assert collection[0] == "This is passage 1"
        assert collection[2] == "This is passage 3"
    
    def test_create_collection_from_file(self):
        """Test creating a collection from a TSV file."""
        # Create a temporary TSV file with the .tsv extension
        fd, path = tempfile.mkstemp(suffix='.tsv')
        temp_file = path
        try:
            with os.fdopen(fd, 'w') as f:
                f.write("0\tThis is passage 1\n")
                f.write("1\tThis is passage 2\n")
                f.write("2\tThis is passage 3\n")
            
            collection = Collection(path=temp_file)
            
            assert len(collection) == 3
            assert collection[0] == "This is passage 1"
            assert collection[2] == "This is passage 3"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)