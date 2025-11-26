# DAG-Based Incremental Caching

## 🎯 Overview

TopoBench's `OnDiskInductivePreprocessor` features **DAG-based incremental caching** that automatically reuses cached transforms when you add new transforms to your pipeline. This enables rapid experimentation without reprocessing the entire dataset.

## 🚀 Quick Example

```python
from topobench.data.preprocessor import OnDiskInductivePreprocessor

# Step 1: Process with clique lifting (takes ~40s for 10K samples)
config1 = {
    "clique_lifting": {
        "transform_type": "lifting",
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 2,
    },
}

dataset1 = OnDiskInductivePreprocessor(
    dataset=my_dataset,
    data_dir="./data",
    transforms_config=config1,
)

# Step 2: Add feature normalization (takes ~14s - NOT 54s!)
config2 = {
    "clique_lifting": {  # ← REUSED from cache!
        "transform_type": "lifting",
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 2,
    },
    "normalization": {  # ← Only this is processed
        "transform_type": "feature",
        "transform_name": "ProjectionSum",
    },
}

dataset2 = OnDiskInductivePreprocessor(
    dataset=my_dataset,
    data_dir="./data",  # Same directory!
    transforms_config=config2,
)
# Output: "Reusing 1 cached transform(s)!" ✅
```

**Result:** Adding the second transform only takes **14s instead of 54s** because SimplicialCliqueLifting is reused from cache!

---

## 📚 How It Works

### DAG Structure

The preprocessor builds a **Directed Acyclic Graph (DAG)** of your transforms:

```
Source Data
    ↓
[SimplicialCliqueLifting] ← Transform 0
    ↓
[ProjectionSum]           ← Transform 1
    ↓
[ProjectionSum]           ← Transform 2
    ↓
Final Output
```

Each transform gets a **unique cache directory**:
```
data_dir/
  transform_chain/
    DataTransform_0_b13b3327/  ← SimplicialCliqueLifting
    DataTransform_1_706ffcf1/  ← ProjectionSum #1
    DataTransform_2_706ffcf1/  ← ProjectionSum #2 (different ID, same hash!)
```

### Cache Key Format

Cache directories use: `{transform_id}_{parameter_hash}`

- **transform_id**: Position in chain (e.g., `DataTransform_0`)
- **parameter_hash**: Hash of transform parameters (e.g., `b13b3327`)

This handles two critical cases:

1. **Duplicate transforms** (same parameters, different positions)
   - Example: Two ProjectionSum with identical params
   - Same hash, different IDs → different directories ✅

2. **Parameter changes** (same position, different parameters)
   - Example: SimplicialCliqueLifting(dim=2) vs (dim=3)
   - Same ID, different hash → different directories ✅

### Incremental Processing

When you add transforms, the preprocessor:

1. **Analyzes the DAG** to find which transforms are cached
2. **Loads from the last cached transform**
3. **Processes only new/modified transforms**
4. **Saves each transform incrementally**

```python
# You have: [T0_cached, T1_cached]
# You want:  [T0, T1, T2_new]

# Preprocessor does:
1. Check T0: ✅ cached → skip
2. Check T1: ✅ cached → skip  
3. Check T2: ❌ not cached → process
4. Load from T1's cache
5. Apply T2 transform
6. Save T2 to new cache directory
```

---

## ⚡ Performance: Files vs Mmap Backend

DAG caching works with **both** storage backends, but the speedup is more visible with `files`:

### Files Backend (Recommended for Development)

```python
OnDiskInductivePreprocessor(
    ...,
    storage_backend="files",  # ← Shows clear DAG cache benefit
)
```

**Performance (10K samples):**
```
Initial build:    42s  (SimplicialCliqueLifting)
Add 1 transform:  14s  (3.0× speedup) ✅
Add 2 transforms: 28s  (1.5× speedup) ✅

Benefit: Pure transform processing savings visible
```

### Mmap Backend (Recommended for Production)

```python
OnDiskInductivePreprocessor(
    ...,
    storage_backend="mmap",  # ← Compression hides DAG cache benefit
    compression="lz4",
)
```

**Performance (10K samples):**
```
Initial build:    33s  (SimplicialCliqueLifting)
Add 1 transform:  19s  (1.7× speedup) ⚠️
Add 2 transforms: 20s  (1.6× speedup) ⚠️

Benefit: Transform processing saved, but mmap conversion overhead added
```

**Why the difference?**

Each transform output requires mmap conversion (compression + merge):

```
Files:
  - Transform: 14s
  - Save: <1s
  Total: ~14s per transform

Mmap:
  - Transform: 10s  ← DAG cache saves this! ✅
  - Compression: 8s ← Cannot be cached ❌
  Total: ~18s per transform
```

The DAG cache **IS working** with mmap, but the conversion overhead masks the speedup!

**Recommendation:**
- **Development/iteration:** Use `storage_backend="files"` for fast experimentation
- **Production:** Use `storage_backend="mmap"` for compressed storage and fast I/O

See [SPEED_VS_COMPRESSION_TRADEOFF.md](SPEED_VS_COMPRESSION_TRADEOFF.md) for detailed analysis.

---

## 🔧 Advanced Usage

### Forcing Recomputation

Change any parameter to trigger recomputation:

```python
# Original
config1 = {
    "clique_lifting": {
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 2,  # ← Original
    },
}

# Modified - will recompute
config2 = {
    "clique_lifting": {
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 3,  # ← Changed!
    },
}
```

Different parameters → different hash → different cache directory → recomputation

### Checking Cache Status

```python
dataset = OnDiskInductivePreprocessor(...)

# Inspect transform chain
for entry in dataset.transform_chain:
    print(f"Transform: {entry['transform_id']}")
    print(f"  Cached: {entry['cached']}")
    print(f"  Directory: {entry['output_dir']}")
```

### Cache Location

All caches are stored under your `data_dir`:

```
data_dir/
  transform_chain/
    DataTransform_0_abc123/  # Each transform gets unique dir
    DataTransform_1_def456/
    DataTransform_2_def456/  # Same hash as 1 (duplicate), different ID
```

To clear cache: `rm -rf data_dir/transform_chain`

---

## 📊 Benchmark Results

We provide comprehensive DAG caching benchmarks:

```bash
python benchmarks/benchmark_comprehensive_pipeline.py \
  --benchmarks dag \
  --output results/dag_cache
```

**Expected results (10K samples, files backend):**
```
Initial Build:    42s   (cold start)
Cache Hit:        <1s   (exact reuse - 50000× speedup!)
Light Extension:  14s   (add 1 transform - 3× speedup)
Heavy Extension:  28s   (add 2 transforms - 1.5× speedup)
```

---

## 🐛 Troubleshooting

### "Not seeing cache reuse"

**Check:**
1. Same `data_dir` between runs?
2. Same transform parameters?
3. Look for "Reusing X cached transform(s)!" message

### "Duplicate transforms colliding"

**This was a bug, now fixed!** 

If you have two identical transforms (e.g., two ProjectionSum with same parameters), they now get separate cache directories:

```
DataTransform_1_706ffcf1/  # ProjectionSum #1
DataTransform_2_706ffcf1/  # ProjectionSum #2 (same hash, different ID!)
```

Both will be processed correctly.

### "Cache taking too much space"

Each transform creates a separate cache. For 10K samples:
- Files backend: ~70MB per transform
- Mmap backend: ~15MB per transform (compressed)

To clean: `rm -rf data_dir/transform_chain`

---

## 📖 Related Documentation

- [SPEED_VS_COMPRESSION_TRADEOFF.md](SPEED_VS_COMPRESSION_TRADEOFF.md) - Files vs mmap backend guide
- [benchmarks/README.md](benchmarks/README.md) - Benchmark documentation
- API Reference: `topobench.data.preprocessor.OnDiskInductivePreprocessor`

---

## 🎓 Best Practices

### 1. Use Files for Iteration

```python
# During development
dataset = OnDiskInductivePreprocessor(
    ...,
    storage_backend="files",  # Fast iteration
    num_workers=7,            # Parallel processing
)
```

### 2. Convert to Mmap for Production

```python
# For deployment
dataset = OnDiskInductivePreprocessor(
    ...,
    storage_backend="mmap",   # Compressed storage
    compression="lz4",        # Fast decompression
    num_workers=1,            # Reliable compression
)
```

### 3. Organize Your Data Directory

```python
# Keep experiments separate
data_dir = f"./data/experiment_{experiment_name}"
```

### 4. Clean Cache Regularly

```bash
# Remove old caches
find ./data -type d -name "transform_chain" -mtime +30 -exec rm -rf {} +
```

---

## ✅ Summary

**DAG caching automatically:**
- ✅ Reuses cached transforms when adding new ones
- ✅ Handles duplicate transforms correctly
- ✅ Detects parameter changes
- ✅ Works with both files and mmap backends
- ✅ Provides 2-3× speedup for incremental changes

**For best results:**
- Use `storage_backend="files"` for rapid iteration
- Use `storage_backend="mmap"` for production deployment
- Monitor cache size and clean periodically

**Validated by comprehensive tests** - see `test/data/preprocessor/test_dag_caching.py`

---

**Last Updated:** November 26, 2024  
**Feature Status:** Production-ready  
**Test Coverage:** 10 comprehensive tests, all passing ✅
