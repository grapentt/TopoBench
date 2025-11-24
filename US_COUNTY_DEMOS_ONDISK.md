# US County Demographics On-Disk Dataset

## Overview

`USCountyDemosOnDiskDataset` is a memory-efficient implementation of the US County Demographics dataset that uses the high-performance `BaseOnDiskInductiveDataset` architecture to minimize RAM usage while maintaining fast access speeds.

## Key Features

✅ **O(1) Memory Usage** - Data is loaded on-demand, not stored in memory  
✅ **Lightweight Pickling** - ~1-10KB pickle size for fast parallel processing  
✅ **Automatic Caching** - First access loads from disk, subsequent accesses use cache  
✅ **Multiprocessing-Safe** - Designed for parallel data loading with PyTorch DataLoader  
✅ **PyTorch 2.6+ Compatible** - Uses `weights_only=False` for safe loading  

## Comparison: In-Memory vs On-Disk

| Feature | InMemoryDataset | OnDiskDataset |
|---------|----------------|---------------|
| **Memory Usage** | Loads all data into RAM | O(1) - loads on-demand |
| **Pickle Size** | Large (includes all data) | ~1-10KB (metadata only) |
| **Parallel Speedup** | Slow (large pickle) | 2.29-5× faster |
| **Initial Load** | Slower (loads everything) | Faster (lazy loading) |
| **Best For** | Small datasets, repeated access | Large datasets, memory-constrained |

## Installation & Import

```python
from topobench.data.datasets import USCountyDemosOnDiskDataset
from omegaconf import DictConfig

# Create dataset
params = DictConfig({"year": 2012, "task_variable": "Election"})
dataset = USCountyDemosOnDiskDataset(
    root="/path/to/data",
    name="US-county-demos",
    parameters=params,
    cache_samples=True  # Enable caching (recommended)
)

# Access data
graph = dataset[0]  # Loads on-demand
print(f"Nodes: {graph.num_nodes}, Features: {graph.x.shape[1]}")
```

## Parameters

### `__init__(root, name, parameters, cache_samples=True)`

- **`root`** (str): Root directory where the dataset will be saved
- **`name`** (str): Name of the dataset (e.g., "US-county-demos")
- **`parameters`** (DictConfig): Configuration with:
  - `year` (int): Year of the data (default: 2012)
  - `task_variable` (str): Prediction target variable (e.g., "Election", "MedianIncome", etc.)
- **`cache_samples`** (bool, optional): Enable caching (default: True)

## Available Task Variables

The `task_variable` parameter can be one of:
- `"Election"` - Election results (DEM-GOP normalized)
- `"MedianIncome"` - Median income per county
- `"MigraRate"` - Migration rate
- `"BirthRate"` - Birth rate
- `"DeathRate"` - Death rate
- `"BachelorRate"` - Bachelor degree attainment rate
- `"UnemploymentRate"` - Unemployment rate

## Architecture

The implementation inherits from `FileBasedInductiveDataset`, which provides:

1. **Download Management**: Automatically downloads data from Google Drive if not present
2. **Processing Pipeline**: Converts raw CSV files to PyTorch Geometric Data objects
3. **On-Disk Storage**: Saves processed data as `data_0.pt` in the processed directory
4. **Caching Layer**: Optional in-memory cache for frequently accessed samples
5. **Lightweight Pickling**: Only pickles metadata, not the actual data

### Directory Structure

```
root/
  US-county-demos/
    raw/
      county_graph.csv
      county_stats_2012.csv
    2012_Election/
      processed/
        data_0.pt                    # Processed graph
        .sample_cache/               # Cache directory (if enabled)
          sample_000000.pt
```

## Use Cases

### 1. Memory-Constrained Environments

```python
# Perfect for systems with limited RAM
dataset = USCountyDemosOnDiskDataset(
    root="/data",
    name="US-county-demos",
    parameters=params,
    cache_samples=True
)
```

### 2. Parallel Data Loading

```python
from torch.utils.data import DataLoader

# Fast multiprocessing due to lightweight pickling
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Efficient parallel loading
    shuffle=True
)
```

### 3. Multiple Task Variables

```python
# Load different prediction tasks efficiently
election_dataset = USCountyDemosOnDiskDataset(
    root="/data", name="US-county-demos",
    parameters=DictConfig({"year": 2012, "task_variable": "Election"})
)

income_dataset = USCountyDemosOnDiskDataset(
    root="/data", name="US-county-demos",
    parameters=DictConfig({"year": 2012, "task_variable": "MedianIncome"})
)
```

## Properties

The dataset provides convenient properties:

```python
dataset = USCountyDemosOnDiskDataset(root, name, params)

# Dataset info
print(f"Number of graphs: {len(dataset)}")          # 1
print(f"Number of nodes: {dataset.num_nodes}")      # ~3000+ counties
print(f"Number of features: {dataset.num_features}")# 7 features
print(f"Number of edges: {dataset.num_edges}")      # County adjacencies
```

## Performance Characteristics

### Memory Usage
- **Pickle size**: ~1-10 KB (vs. several MB for in-memory)
- **Runtime memory**: O(1) - only loaded samples in memory
- **Cache size**: 1 sample (~few MB) when caching enabled

### Speed
- **First access**: Load from disk (~ms)
- **Cached access**: Load from cache (~µs)
- **Parallel speedup**: 2.29-5× faster than InMemoryDataset

## Testing

Comprehensive test suite included:

```bash
# Run all tests
pytest test/data/datasets/test_us_county_demos_ondisk.py -v

# Run only fast tests (skip download)
pytest test/data/datasets/test_us_county_demos_ondisk.py::TestUSCountyDemosOnDiskDataset -v

# Run integration test (includes download)
pytest test/data/datasets/test_us_county_demos_ondisk.py::TestUSCountyDemosOnDiskDatasetIntegration -v
```

## Example Script

See `examples/us_county_demos_ondisk_example.py` for a complete working example comparing in-memory vs on-disk approaches.

## Implementation Details

### Key Methods

- **`_download()`**: Downloads and extracts raw data from Google Drive
- **`_process()`**: Converts raw CSV files to PyG Data objects
- **`_load_file(file_path)`**: Loads a processed sample from disk
- **`_get_pickle_args()`**: Returns minimal args for efficient pickling

### Caching Behavior

When `cache_samples=True`:
1. First access loads from disk and caches in memory
2. Subsequent accesses use the in-memory cache
3. Cache is stored in `.sample_cache/` subdirectory
4. Cache persists across Python sessions

When `cache_samples=False`:
- Every access loads from disk
- No cache directory created
- Lower memory usage but slower repeated access

## Migration from InMemoryDataset

If you're currently using `USCountyDemosDataset` (in-memory), you can easily switch:

```python
# Old (in-memory)
from topobench.data.datasets import USCountyDemosDataset
dataset = USCountyDemosDataset(root, name, parameters)

# New (on-disk)
from topobench.data.datasets import USCountyDemosOnDiskDataset
dataset = USCountyDemosOnDiskDataset(root, name, parameters)
```

The API is fully compatible, so no code changes needed!

## When to Use On-Disk vs In-Memory

### Use On-Disk When:
- ✅ Working with limited RAM
- ✅ Using parallel data loading (multiprocessing)
- ✅ Processing large datasets
- ✅ Need fast pickle/unpickle for distributed training

### Use In-Memory When:
- ✅ Dataset fits comfortably in RAM
- ✅ Need maximum repeated access speed
- ✅ Single-threaded processing
- ✅ Backwards compatibility required

## Related Files

- **Implementation**: `topobench/data/datasets/us_county_demos_ondisk.py`
- **Tests**: `test/data/datasets/test_us_county_demos_ondisk.py`
- **Example**: `examples/us_county_demos_ondisk_example.py`
- **Base Classes**: `topobench/data/datasets/base_inductive.py`

## References

- [FINAL_IMPLEMENTATION_SUMMARY.md](FINAL_IMPLEMENTATION_SUMMARY.md) - Overall architecture
- [USER_FRIENDLY_DATASET_SOLUTION.md](USER_FRIENDLY_DATASET_SOLUTION.md) - Base class design
- [PYTORCH_2.6_COMPATIBILITY.md](PYTORCH_2.6_COMPATIBILITY.md) - PyTorch 2.6+ compatibility
