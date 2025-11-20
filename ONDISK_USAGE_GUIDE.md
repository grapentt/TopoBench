# 📚 OnDisk vs InMemory: Usage Guide

## When to Use OnDiskInductiveDataset

### ✅ Use OnDisk When:

1. **Dataset Size > Available RAM**
   - Your dataset + lifted features exceed available memory
   - Example: 100K graphs with simplicial lifting = several GB
   - **OnDisk Solution:** Constant O(1) memory usage

2. **Memory-Intensive Transforms**
   - Lifting operations (graph → simplicial/cell/hypergraph)
   - Feature engineering that expands data significantly
   - **OnDisk Solution:** Process one sample at a time

3. **Iterative Experimentation**
   - Testing multiple models on same preprocessed data
   - Hyperparameter tuning across many runs
   - **OnDisk Solution:** Cache preprocessing, instant reload

4. **Production Deployments**
   - Need predictable memory footprint
   - Running on memory-constrained hardware
   - **OnDisk Solution:** Scales to any dataset size

5. **Large-Scale Research**
   - Benchmark suites with many datasets
   - Reproducibility across different machines
   - **OnDisk Solution:** Consistent behavior regardless of RAM

### ❌ Use InMemory (PreProcessor) When:

1. **Small Datasets**
   - Dataset fits comfortably in RAM (<1GB total)
   - Example: MUTAG (188 graphs), PROTEINS (1113 graphs)
   - **InMemory Advantage:** Slightly faster (no disk I/O)

2. **One-Time Experiments**
   - Quick prototyping, won't reuse processed data
   - Single training run
   - **InMemory Advantage:** Simpler (no cache management)

3. **No Transforms**
   - Using data as-is without lifting
   - Minimal feature engineering
   - **InMemory Advantage:** Traditional approach, well-tested

4. **Abundant RAM Available**
   - Server with 128GB+ RAM
   - Dataset + features < 10% of available RAM
   - **InMemory Advantage:** Maximum speed

---

## Performance Comparison

| Aspect | OnDisk | InMemory (PreProcessor) |
|--------|--------|-------------------------|
| **Memory Usage** | O(1) constant | O(N) grows with dataset |
| **First Run Speed** | ~200 samples/sec | ~200 samples/sec |
| **Cached Run Speed** | Instant (<0.1 sec) | N/A (no cache) |
| **Max Dataset Size** | Unlimited (disk-bound) | RAM-bound |
| **Disk Usage** | ~file per sample | Single collated file |
| **Complexity** | Moderate (caching) | Simple (direct) |

---

## Usage Examples

### Basic Usage (No Transforms)

```python
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductiveDataset
from torch_geometric.datasets import TUDataset

# Load source dataset
source = TUDataset(root='/data/source', name='ENZYMES')

# Create OnDisk dataset (no transforms)
dataset = OnDiskInductiveDataset(
    dataset=source,
    data_dir='/data/processed',
    transforms_config=None,  # No transforms
)

# Use in training
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    # Train model
    pass
```

### With Transforms (Simplicial Lifting)

```python
from omegaconf import DictConfig

# Configure lifting transform
transform_config = DictConfig({
    'transform_name': 'SimplicialCliqueLifting',
    'complex_dim': 3,
    'signed': False,
})

# Create OnDisk dataset with transforms
dataset = OnDiskInductiveDataset(
    dataset=source,
    data_dir='/data/processed',
    transforms_config=transform_config,
)

# First run: processes all samples (takes time)
# Subsequent runs: instant cache hit!
```

### Integration with TopoBench Loader

```python
from topobench.data.loaders.base import AbstractLoader
from omegaconf import DictConfig

class OnDiskTUDatasetLoader(AbstractLoader):
    """Custom loader with OnDisk backend."""
    
    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.transforms_config = parameters.get("transforms", None)
    
    def load_dataset(self):
        """Load dataset with OnDisk backend."""
        # Load source dataset
        source = TUDataset(
            root=str(self.root_data_dir / "source"),
            name=self.parameters.data_name,
        )
        
        # Wrap with OnDiskInductiveDataset
        ondisk = OnDiskInductiveDataset(
            dataset=source,
            data_dir=self.root_data_dir / "processed",
            transforms_config=self.transforms_config,
        )
        
        return ondisk

# Use in Hydra pipeline
loader = hydra.utils.instantiate(cfg.dataset.loader)
dataset, data_dir = loader.load()

# Load splits (same as standard)
train, val, test = dataset.load_dataset_splits(cfg.dataset.split_params)

# Create dataloader (same as standard)
dataloader = TBDataloader(train, val, test, **cfg.dataset.dataloader_params)

# Train (same as standard)
trainer.fit(model, dataloader)
```

### Force Reload (Refresh Cache)

```python
# Reprocess all samples (useful after code changes)
dataset = OnDiskInductiveDataset(
    dataset=source,
    data_dir='/data/processed',
    transforms_config=transform_config,
    force_reload=True,  # Ignore cache
)
```

### Custom Transform Pipeline

```python
# Multiple transforms in sequence
transform_config = DictConfig({
    'lifting1': {
        'transform_name': 'SimplicialCliqueLifting',
        'complex_dim': 2,
    },
    'lifting2': {
        'transform_name': 'NodeFeatureAugmentation',
        'method': 'one_hot_degree',
    },
})

dataset = OnDiskInductiveDataset(
    dataset=source,
    data_dir='/data/processed',
    transforms_config=transform_config,
)

# Different configs → different cache directories
# Experiment freely without reprocessing!
```

---

## Advanced Topics

### Cache Directory Structure

```
data_dir/
├── no_transforms/                    # No transforms applied
│   ├── sample_000000.pt
│   ├── sample_000001.pt
│   ├── ...
│   └── metadata.json
│
├── transform_name_complex_dim/       # Transform with parameters
│   ├── 2937018487/                   # Parameter hash
│   │   ├── sample_000000.pt
│   │   ├── ...
│   │   └── metadata.json
│   └── 3050422370/                   # Different hash (different params)
│       ├── sample_000000.pt
│       ├── ...
│       └── metadata.json
```

### Parameter Hashing

OnDisk uses deterministic hashing to create unique cache directories:

```python
# Config 1: complex_dim=2
config1 = DictConfig({'transform_name': 'SimplicialCliqueLifting', 'complex_dim': 2})
# → Hash: 2937018487

# Config 2: complex_dim=3 (different!)
config2 = DictConfig({'transform_name': 'SimplicialCliqueLifting', 'complex_dim': 3})
# → Hash: 3050422370 (different directory)

# Same config → same hash → reuses cache
config3 = DictConfig({'transform_name': 'SimplicialCliqueLifting', 'complex_dim': 2})
# → Hash: 2937018487 (same as config1, cache hit!)
```

### Memory Profile

```python
import tracemalloc

# Start memory tracking
tracemalloc.start()

# Create OnDisk dataset (sequential processing)
dataset = OnDiskInductiveDataset(
    dataset=source,
    data_dir='/data/processed',
    transforms_config=transform_config,
)

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024**2:.1f} MB")
print(f"Peak: {peak / 1024**2:.1f} MB")

# Expected: Peak ≈ single sample size (not dataset size!)
```

### Troubleshooting

**Problem: "Sample file not found" error**
```python
# Solution: Force reload to recreate cache
dataset = OnDiskInductiveDataset(..., force_reload=True)
```

**Problem: Different results with same config**
```python
# Check: Are transform parameters identical?
print(dataset.transforms_parameters)

# Solution: Ensure all parameters are specified consistently
# Random seeds, floating point values, etc.
```

**Problem: Slow processing speed**
```python
# Causes:
# 1. Complex transforms (expected)
# 2. Slow disk (use SSD if possible)
# 3. Network storage (prefer local disk)

# Workaround: Process once, then use cache forever
```

**Problem: Disk space concerns**
```python
# Each sample: ~10-50 KB (graphs) to 1-10 MB (large graphs with features)
# Estimate: samples × avg_size

# Solution: Clean old caches
import shutil
shutil.rmtree('/data/processed/old_experiment/')
```

---

## Migration Guide

### From PreProcessor to OnDisk

**Before (PreProcessor):**
```python
from topobench.data.preprocessor import PreProcessor

preprocessor = PreProcessor(dataset, data_dir, transforms_config)
train, val, test = preprocessor.load_dataset_splits(split_params)
```

**After (OnDisk):**
```python
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductiveDataset

ondisk = OnDiskInductiveDataset(dataset, data_dir, transforms_config)
train, val, test = ondisk.load_dataset_splits(split_params)
```

**That's it!** Just change the class. Everything else stays the same.

---

## Best Practices

### 1. Use Descriptive data_dir Names
```python
# ✅ Good
data_dir = f'/data/experiments/{experiment_name}/{dataset_name}'

# ❌ Bad
data_dir = '/data/temp'
```

### 2. Version Your Transforms
```python
# Include version in config for reproducibility
config = DictConfig({
    'transform_name': 'SimplicialCliqueLifting',
    'complex_dim': 2,
    'version': 'v1',  # Document transform version
})
```

### 3. Monitor First Run
```python
# OnDisk shows progress bar
# Watch for:
# - Processing speed (~200 samples/sec expected)
# - Memory usage (should stay constant)
# - Disk space (check available space)
```

### 4. Clean Up Old Experiments
```python
# Periodically remove unused caches
import shutil
from pathlib import Path

cache_dir = Path('/data/processed')
for exp_dir in cache_dir.iterdir():
    if exp_dir.stat().st_mtime < threshold:  # Old cache
        shutil.rmtree(exp_dir)
```

### 5. Document Your Experiments
```python
# Save experiment metadata
metadata = {
    'dataset': 'ENZYMES',
    'transform': 'SimplicialCliqueLifting',
    'date': '2025-11-20',
    'cache_dir': str(dataset.processed_dir),
}

with open('/data/experiments/log.json', 'w') as f:
    json.dump(metadata, f)
```

---

## FAQ

**Q: Can I use OnDisk with transductive learning?**  
A: OnDisk is designed for inductive learning (separate train/val/test samples). For transductive learning (single graph), see the transductive loader (Mission 2/B1 Bonus).

**Q: Does OnDisk work with custom datasets?**  
A: Yes! Any dataset that implements `__len__` and `__getitem__` works. PyG datasets, PyTorch datasets, or custom classes.

**Q: Can I modify samples after caching?**  
A: Use `force_reload=True` to reprocess with new logic. Or manually delete cache directory.

**Q: Is OnDisk faster than InMemory?**  
A: First run: same speed. Subsequent runs: OnDisk is 20-150x faster due to caching. During training: similar speed (disk I/O is async).

**Q: How much disk space do I need?**  
A: Rule of thumb: 2-5x dataset RAM requirements. Example: 10GB dataset → 20-50GB disk.

**Q: Can multiple processes share a cache?**  
A: Yes! OnDisk reads are safe. Avoid concurrent writes (only one process should create cache).

**Q: Does it work on network storage?**  
A: Yes, but slower. Local SSD recommended for best performance.

---

## Summary

**Choose OnDisk when:**
- Dataset > RAM
- Using memory-intensive transforms
- Running multiple experiments
- Need reproducibility
- Production deployment

**Choose InMemory when:**
- Small dataset (<1GB)
- One-time experiment
- Abundant RAM available
- Simplicity preferred

**Migration is easy:** Just swap `PreProcessor` → `OnDiskInductiveDataset`. Everything else stays the same!

---

**For more details, see:**
- `topobench/data/preprocessor/ondisk_inductive.py` - Implementation
- `test/data/preprocessor/test_ondisk_inductive.py` - Comprehensive tests
- `MASTER_PLAN.md` - Architecture and design decisions
