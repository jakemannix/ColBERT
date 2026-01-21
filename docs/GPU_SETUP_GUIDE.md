# GPU Setup Guide for ColBERT with Filtered Search

This guide covers setting up ColBERT with GPU-accelerated filtered search on A100/H100 GPUs using Docker.

## Quick Start

```bash
# Build the Docker image
docker build -t colbert-gpu -f docker/Dockerfile.gpu .

# Run with GPU access
docker run --gpus all -it -v $(pwd)/data:/app/data colbert-gpu

# Run benchmarks
python tests/benchmark_gpu_filtering.py
```

---

## Prerequisites

### Hardware Requirements
- NVIDIA A100 (40GB or 80GB) or H100 GPU
- Minimum 32GB system RAM (64GB+ recommended for large collections)
- Fast NVMe storage for index files

### Software Requirements
- Docker 24.0+ with NVIDIA Container Toolkit
- NVIDIA Driver 525+ (for CUDA 12.x support)

### Cloud Provider Options
- **AWS**: p4d.24xlarge (A100), p5.48xlarge (H100)
- **GCP**: a2-highgpu-1g (A100), a3-highgpu-8g (H100)
- **Azure**: Standard_NC24ads_A100_v4, Standard_ND96isr_H100_v5
- **Lambda Labs**: gpu_1x_a100, gpu_1x_h100_pcie

---

## Docker Setup

### Option 1: Use Pre-built NVIDIA Base Image (Recommended)

Create `docker/Dockerfile.gpu`:

```dockerfile
# Use NVIDIA PyTorch container as base (includes CUDA, cuDNN, PyTorch)
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install faiss-gpu (built for CUDA 12)
RUN pip install faiss-gpu

# Copy ColBERT source
COPY . /app/

# Install ColBERT and dependencies
RUN pip install -e ".[dev]"

# Install additional dependencies for GPU filtering
RUN pip install pytest pytest-benchmark

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["bash"]
```

### Option 2: Build from CUDA Base Image

For more control over versions:

```dockerfile
# docker/Dockerfile.gpu-custom
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system deps
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install faiss-gpu
RUN pip install faiss-gpu

WORKDIR /app
COPY . /app/

# Install ColBERT
RUN pip install -e ".[dev]"

# Install benchmark dependencies
RUN pip install pytest pytest-benchmark

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
```

---

## Building and Running

### Build the Image

```bash
cd /path/to/ColBERT

# Using the recommended Dockerfile
docker build -t colbert-gpu -f docker/Dockerfile.gpu .
```

### Run Interactive Shell

```bash
# Single GPU
docker run --gpus all -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/experiments:/app/experiments \
    colbert-gpu bash

# Specific GPU
docker run --gpus '"device=0"' -it \
    -v $(pwd)/data:/app/data \
    colbert-gpu bash

# Multiple GPUs
docker run --gpus '"device=0,1"' -it \
    -v $(pwd)/data:/app/data \
    colbert-gpu bash
```

### Run with Shared Memory (Required for Large Collections)

```bash
docker run --gpus all -it \
    --shm-size=32g \
    -v $(pwd)/data:/app/data \
    colbert-gpu bash
```

---

## Verification

### Check GPU Access

```bash
# Inside the container
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA A100-SXM4-80GB
CUDA version: 12.1
```

### Run Unit Tests

```bash
# Run GPU filtering tests
python -m pytest tests/test_gpu_filtering.py -v

# Expected: All tests pass (will use CUDA if available)
```

### Run Benchmarks

```bash
# Full benchmark suite
python tests/benchmark_gpu_filtering.py

# Quick smoke test
python -c "
from tests.benchmark_gpu_filtering import SyntheticCollection
from colbert.search.gpu_metadata import GPUMetadataStore

# Create small test collection
collection = SyntheticCollection(n_docs=10000)

# Build metadata store on GPU
store = GPUMetadataStore.from_collection(
    collection,
    {'category': 'categorical', 'year': 'int', 'tags': 'multi_categorical'},
    device='cuda'
)
print(f'Built store with {store.n_docs} docs on {store.device}')
print(f'Memory usage: {store.memory_usage() / 1e6:.2f} MB')
"
```

---

## Cloud-Specific Instructions

### AWS EC2 (p4d/p5 instances)

```bash
# Launch instance with Deep Learning AMI
# AMI: Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)

# Install Docker and NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y docker.io
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Clone and build
git clone https://github.com/stanford-futuredata/ColBERT.git
cd ColBERT
docker build -t colbert-gpu -f docker/Dockerfile.gpu .
docker run --gpus all -it --shm-size=32g colbert-gpu
```

### GCP (a2/a3 instances)

```bash
# Use Deep Learning VM image
# Image family: pytorch-latest-gpu

# Install NVIDIA Container Toolkit
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Build and run
git clone https://github.com/stanford-futuredata/ColBERT.git
cd ColBERT
docker build -t colbert-gpu -f docker/Dockerfile.gpu .
docker run --gpus all -it --shm-size=32g colbert-gpu
```

### Lambda Labs

```bash
# Lambda instances come with Docker and nvidia-docker pre-installed
git clone https://github.com/stanford-futuredata/ColBERT.git
cd ColBERT
docker build -t colbert-gpu -f docker/Dockerfile.gpu .
docker run --gpus all -it --shm-size=32g colbert-gpu
```

---

## Running GPU-Accelerated Filtered Search

### Example: Index and Search with Filtering

```python
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from colbert.search import FilteredIndexScorer, create_filtered_searcher

# 1. Index your collection (standard ColBERT indexing)
with Run().context(RunConfig(nranks=1, experiment="filtered_search")):
    config = ColBERTConfig(nbits=2, root="/app/experiments")
    indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
    indexer.index(name="my_index", collection="/app/data/collection.tsv")

# 2. Create filtered searcher with GPU metadata
searcher = create_filtered_searcher(
    index_path="/app/experiments/indexes/my_index",
    collection=collection,  # Your collection with metadata
    fields_config={
        'category': 'categorical',
        'year': 'int',
        'tags': 'multi_categorical',
    },
    use_gpu=True
)

# 3. Search with GPU-accelerated filtering
pids, ranks, scores = searcher.search(
    "quantum computing applications",
    k=10,
    filters={
        "category": "science",
        "year": ">=2020",
        "tags": ["physics", "quantum"]  # HAS_ANY
    }
)

print(f"Top results: {pids}")
print(f"Scores: {scores}")
```

### Benchmark Filtered Search Performance

```bash
# Run the full benchmark suite
python tests/benchmark_gpu_filtering.py

# Output includes:
# - Build time for metadata store at various scales
# - Filter application latency by candidate count
# - Cache hit vs miss performance
# - Memory usage statistics
```

---

## Performance Tuning

### Memory Optimization

```python
# For very large collections, adjust cache settings
scorer = FilteredIndexScorer(
    index_path=index_path,
    collection=collection,
    fields_config=fields_config,
    use_gpu=True,
    cache_config={
        'max_entries': 500,           # Reduce cache entries
        'max_memory_bytes': 2e9,      # 2GB cache budget
        'ttl_seconds': 1800,          # 30 min TTL
        'min_candidates': 512,        # Increase min candidates
        'max_ncells': 64,             # Allow more centroid probing
        'selectivity_threshold': 0.0005  # More aggressive brute-force
    }
)
```

### Multi-GPU Setup

For very large collections, you can shard the index:

```python
# Index with multiple GPUs
with Run().context(RunConfig(nranks=4, experiment="large_index")):
    config = ColBERTConfig(nbits=2)
    indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
    indexer.index(name="large_index", collection="/app/data/large_collection.tsv")
```

### A100 vs H100 Considerations

| Aspect | A100 (80GB) | H100 (80GB) |
|--------|-------------|-------------|
| Memory Bandwidth | 2 TB/s | 3.35 TB/s |
| Tensor Cores | 432 | 528 |
| Best for | Large batch filtering | Real-time filtering |
| Expected speedup | 1x baseline | ~1.5-2x for memory-bound ops |

The GPU filtering operations are primarily memory-bound (loading posting lists, tensor comparisons), so H100's higher memory bandwidth provides significant benefits.

---

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size for filtering
scorer = FilteredIndexScorer(
    ...,
    cache_config={'max_memory_bytes': 1e9}  # 1GB limit
)

# Or process in smaller chunks
chunk_size = 100000
for i in range(0, len(all_pids), chunk_size):
    chunk_pids = all_pids[i:i+chunk_size]
    filtered = gpu_filter.apply(chunk_pids, filters)
```

### Slow First Query

The first query may be slow due to:
1. JIT compilation of CUDA kernels
2. Cold metadata cache
3. Model loading

Run a warmup query:
```python
# Warmup
_ = searcher.search("warmup query", k=1)

# Now benchmark
import time
start = time.time()
results = searcher.search("real query", k=100, filters={"category": "science"})
print(f"Query time: {time.time() - start:.3f}s")
```

### Docker GPU Not Detected

```bash
# Check nvidia-docker installation
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Version Compatibility Matrix

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| Python | 3.10 | 3.8 |
| PyTorch | 2.1+ | 1.13 |
| CUDA | 12.1 | 11.8 |
| NVIDIA Driver | 530+ | 525 |
| faiss-gpu | 1.7.4+ | 1.7.0 |
| transformers | 4.35+ | 4.20 |

---

## Next Steps

1. **Run the benchmarks** to establish baseline performance on your hardware
2. **Index your collection** with metadata fields
3. **Test filtered search** with representative queries
4. **Monitor cache hit rates** and adjust configuration as needed

For questions or issues, open an issue at: https://github.com/stanford-futuredata/ColBERT/issues
