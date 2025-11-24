# Dataset Download Patterns in TopoBench

This guide explains different approaches for implementing dataset downloads, with memory efficiency considerations.

## Overview

Most datasets implement downloading in the `download()` method of PyG's `InMemoryDataset` or custom dataset classes. The key consideration is whether to use **in-memory** or **streaming** downloads.

## Pattern 1: In-Memory Download (Standard)

### Example from US County Demos Dataset

```python
from torch_geometric.data import InMemoryDataset
from topobench.data.utils import download_file_from_drive

class USCountyDemosDataset(InMemoryDataset):
    URLS = {
        "US-county-demos": "https://drive.google.com/file/d/1FNF_LbByhY.../view"
    }
    
    def download(self):
        # Download file (loaded into memory during download)
        download_file_from_drive(
            file_link=self.URLS[self.name],
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format="zip"
        )
        
        # Extract (also uses memory)
        zip_path = osp.join(self.raw_dir, f"{self.name}.zip")
        extract_zip(zip_path, self.raw_dir)
        os.remove(zip_path)  # Clean up
```

### When to Use

✅ **Most datasets** - This is the standard approach
- File size < 10 GB
- Download is one-time operation
- Simpler implementation

### Memory Usage

- **During download**: ~File size in RAM
- **After download**: 0 (data saved to disk)

**Key insight**: Download memory usage is temporary and happens only once. The real bottleneck for large-scale training is **preprocessing**, not downloading.

---

## Pattern 2: Streaming Download (Advanced)

For very large files that exceed available RAM, implement chunk-based streaming:

```python
import requests
from pathlib import Path

def streaming_download(url: str, output_path: Path, chunk_size: int = 8192):
    """Download large file in chunks without loading entire file into memory.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path
        Where to save the file
    chunk_size : int
        Size of chunks to download (default: 8KB)
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # Filter out keep-alive chunks
                f.write(chunk)

class VeryLargeDataset(InMemoryDataset):
    def download(self):
        output_path = Path(self.raw_dir) / "large_file.bin"
        
        # Stream download - never loads full file into RAM
        streaming_download(self.url, output_path, chunk_size=8192)
        
        # Process in chunks if needed
        with open(output_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                process_chunk(chunk)
```

### When to Use

✅ **Very large files** (> 10 GB)
✅ **Limited RAM** environments
✅ **Cloud storage** with streaming APIs

### Memory Usage

- **During download**: ~chunk_size (e.g., 8 KB)
- **Peak RAM**: Minimal, independent of file size

---

## Pattern 3: Direct Integration with BaseOnDiskInductiveDataset

For maximum memory efficiency, combine on-disk dataset with streaming download:

```python
from topobench.data.datasets import OnDemandInductiveDataset
import numpy as np

class StreamedSubgraphDataset(OnDemandInductiveDataset):
    """Download large graph, store as memory-mapped array, extract on-demand."""
    
    def __init__(self, root, num_samples, url):
        self.url = url
        
        # Download if needed (could use streaming)
        self._download_if_needed()
        
        # Load as memory-mapped array (not into RAM!)
        self.edges_mmap = np.load(
            Path(root) / "edges.npy", 
            mmap_mode='r'  # Read-only, stays on disk
        )
        self.features_mmap = np.load(
            Path(root) / "features.npy",
            mmap_mode='r'
        )
        
        super().__init__(root, num_samples, seed=42)
    
    def _download_if_needed(self):
        edges_path = Path(self.root) / "edges.npy"
        if not edges_path.exists():
            # Use streaming download for large files
            streaming_download(self.url, edges_path)
    
    def _generate_sample(self, idx, rng):
        # Extract subgraph on-demand from memory-mapped data
        # Only loads the needed portion into RAM
        target_node = idx % len(self.features_mmap)
        neighborhood = self._extract_k_hop(target_node, k=2)
        
        # Load only needed features (not entire array!)
        x = torch.from_numpy(self.features_mmap[neighborhood].copy())
        return Data(x=x, ...)
```

### Memory Profile

| Stage | RAM Usage |
|-------|-----------|
| **Download** | ~8 KB (if streaming) |
| **Storage** | 0 (memory-mapped) |
| **Sample access** | O(subgraph size) |

**Example**: Papers100M dataset
- Full graph: 44 GB
- Memory-mapped: 0 MB in RAM
- Sample access: ~10 MB per subgraph

---

## Comparison: Download Strategies

| Strategy | RAM During Download | Implementation Complexity | When to Use |
|----------|---------------------|---------------------------|-------------|
| **In-memory** | ~File size | Simple | Most datasets (< 10 GB) |
| **Streaming** | ~Chunk size (KB) | Medium | Very large files (> 10 GB) |
| **Memory-mapped** | ~0 | Complex | Massive graphs (Papers100M) |

---

## Best Practices

### 1. For Most Datasets: Use In-Memory Download

```python
class StandardDataset(InMemoryDataset):
    def download(self):
        # Simple and sufficient for most cases
        download_file_from_drive(self.url, self.raw_dir, ...)
        extract_zip(...)
```

**Rationale**: 
- Download is one-time cost
- Preprocessing is the real memory bottleneck
- Simpler code = fewer bugs

### 2. For Very Large Files: Add Streaming

```python
class LargeDataset(InMemoryDataset):
    def download(self):
        if file_size > 10_000_000_000:  # > 10 GB
            streaming_download(self.url, self.raw_dir)
        else:
            download_file_from_drive(self.url, self.raw_dir)
```

### 3. For Massive Graphs: Combine On-Demand + Memory-Mapping

```python
class MassiveGraphDataset(OnDemandInductiveDataset):
    def __init__(self, root, num_samples):
        self._download_if_needed()
        # Use memory-mapped arrays
        self.data_mmap = np.load(..., mmap_mode='r')
        super().__init__(root, num_samples)
    
    def _generate_sample(self, idx, rng):
        # Extract portions on-demand
        ...
```

---

## Example: US County Demos Dataset

The US County Demos dataset demonstrates the standard pattern:

```python
class USCountyDemosDataset(InMemoryDataset):
    def download(self):
        # 1. Download from Google Drive
        download_file_from_drive(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format="zip"
        )
        
        # 2. Extract zip file
        zip_path = osp.join(self.raw_dir, f"{self.name}.zip")
        extract_zip(zip_path, self.raw_dir)
        
        # 3. Clean up
        os.remove(zip_path)
```

**Memory usage**:
- Download: ~50 MB (temporary)
- After download: 0 MB
- Preprocessing: Constant O(1) with OnDiskInductivePreprocessor

This is perfectly fine because:
1. 50 MB is manageable
2. Preprocessing uses O(1) memory with on-disk approach
3. Training uses O(batch_size) memory

---

## Summary

### Key Takeaway

**For 95% of datasets**: Standard in-memory download is fine. The memory bottleneck is in **preprocessing and training**, not downloading.

**Memory-efficient approach**:
1. ✅ Download with standard methods (in-memory is OK)
2. ✅ Use `OnDiskInductivePreprocessor` for preprocessing
3. ✅ Use `BaseOnDiskInductiveDataset` implementations for training
4. ✅ Result: O(1) memory for the entire pipeline

**Only use streaming download if**:
- File size > 10 GB AND
- Available RAM < File size

### Priority for Memory Efficiency

1. **First**: Use on-disk preprocessing (OnDiskInductivePreprocessor)
2. **Second**: Use lightweight datasets (BaseOnDiskInductiveDataset)
3. **Last**: Optimize download (only if file is massive)

The first two steps give you **10-100× memory reduction**. Download optimization gives you ~1-2× reduction and only matters for very large files.
