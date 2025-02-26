
# Could be .tsv or .json. The latter always allows more customization via optional parameters.
# I think it could be worth doing some kind of parallel reads too, if the file exceeds 1 GiBs.
# Just need to use a datastructure that shares things across processes without too much pickling.
# I think multiprocessing.Manager can do that!

import os
import itertools

from colbert.evaluation.loaders import load_collection
from colbert.infra.run import Run


class Collection:
    def __init__(self, path=None, data=None, metadata=None):
        self.path = path
        self.metadata = {} if metadata is None else metadata
        
        if data is not None:
            self.data = data
        elif path is not None:
            self.data = self._load_file(path)
        else:
            # If both path and data are None, initialize with empty list
            self.data = []

    def __iter__(self):
        # TODO: If __data isn't there, stream from disk!
        return self.data.__iter__()

    def __getitem__(self, item):
        # TODO: Load from disk the first time this is called. Unless self.data is already not None.
        return self.data[item]
        
    def get_metadata(self, pid):
        """
        Get metadata for a specific passage by PID.
        
        Args:
            pid: The passage ID
            
        Returns:
            Dict of metadata fields or empty dict if no metadata exists
        """
        return self.metadata.get(pid, {})

    def __len__(self):
        # TODO: Load here too. Basically, let's make data a property function and, on first call, either load or get __data.
        return len(self.data)

    def _load_file(self, path):
        self.path = path
        return self._load_tsv(path) if path.endswith('.tsv') else self._load_jsonl(path)

    def _load_tsv(self, path):
        return load_collection(path)

    def _load_jsonl(self, path):
        """
        Load collection from JSONL file with metadata support.
        Expected format for each line:
        {"pid": 0, "passage": "text", "metadata": {"field1": "value1", "field2": "value2"}}
        """
        import json
        passages = []
        metadata = {}
        
        with open(path, 'r') as f:
            for line_idx, line in enumerate(f):
                if line_idx % (1000*1000) == 0:
                    print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)
                
                try:
                    record = json.loads(line.strip())
                    
                    # Ensure required fields are present
                    pid = record.get('pid', line_idx)
                    passage = record.get('passage', '')
                    
                    # Add to collection
                    passages.append(passage)
                    
                    # Process metadata if present
                    if 'metadata' in record and isinstance(record['metadata'], dict):
                        metadata[pid] = record['metadata']
                        
                except json.JSONDecodeError:
                    # Skip malformed lines
                    print(f"Warning: Skipping malformed JSONL line {line_idx}")
                except Exception as e:
                    print(f"Error processing line {line_idx}: {e}")
                    
        # Store metadata in the object
        self.metadata = metadata
        
        return passages

    def provenance(self):
        return self.path
    
    def toDict(self):
        return {
            'provenance': self.provenance(),
            'has_metadata': len(self.metadata) > 0
        }

    def save(self, new_path):
        assert new_path.endswith('.tsv') or new_path.endswith('.jsonl'), "Only .tsv and .jsonl formats are supported"
        assert not os.path.exists(new_path), new_path

        if new_path.endswith('.tsv'):
            # Save in TSV format (without metadata)
            with Run().open(new_path, 'w') as f:
                for pid, content in enumerate(self.data):
                    content = f'{pid}\t{content}\n'
                    f.write(content)
                
                return f.name
        else:
            # Save in JSONL format with metadata
            import json
            with Run().open(new_path, 'w') as f:
                for pid, content in enumerate(self.data):
                    record = {
                        'pid': pid,
                        'passage': content
                    }
                    
                    # Include metadata if it exists for this passage
                    if pid in self.metadata:
                        record['metadata'] = self.metadata[pid]
                    
                    f.write(json.dumps(record) + '\n')
                
                return f.name

    def enumerate(self, rank):
        for _, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (offset + idx, passage)

    def enumerate_batches(self, rank, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(Run().nranks))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return
    
    def get_chunksize(self):
        return min(25_000, 1 + len(self) // Run().nranks)  # 25k is great, 10k allows things to reside on GPU??

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if type(obj) is list:
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"


# TODO: Look up path in some global [per-thread or thread-safe] list.
