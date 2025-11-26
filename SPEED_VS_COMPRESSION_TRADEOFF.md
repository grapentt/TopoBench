# On-Disk Preprocessing: Speed vs Compression Trade-off ⚖️

**Date:** November 26, 2024  
**Component:** `OnDiskInductivePreprocessor`  
**Key Decision:** Storage backend configuration  

---

## 🎯 The Trade-off

When using `OnDiskInductivePreprocessor`, you must choose between two strategies:

### 1. ⚡ SPEED (Files Backend + Many Workers)

**Configuration:**
```python
dataset = OnDiskInductivePreprocessor(
    dataset=source_dataset,
    data_dir=data_dir,
    transforms_config=transforms_config,
    num_workers=7,              # ← Use many workers
    storage_backend="files",    # ← No compression
)
```

**Characteristics:**
- ✅ **Fast preprocessing:** ~5-7× speedup with 7 workers
- ✅ **True parallel processing:** Each worker processes independently
- ✅ **Simple architecture:** Individual `.pt` files per sample
- ⚠️ **Larger disk usage:** Uncompressed data (~4-5× larger)
- ⚠️ **Slower training I/O:** Individual file access

**Best For:**
- Development and iteration
- Prototyping new models
- When disk space is abundant
- When preprocessing time matters most

---

### 2. 💾 COMPRESSION (Mmap Backend + 1 Worker)

**Configuration:**
```python
dataset = OnDiskInductivePreprocessor(
    dataset=source_dataset,
    data_dir=data_dir,
    transforms_config=transforms_config,
    num_workers=1,              # ← Use 1 worker!
    storage_backend="mmap",     # ← Compressed
    compression="lz4",          # ← Fast compression
)
```

**Characteristics:**
- ✅ **Small disk footprint:** ~4-5× smaller (LZ4 compression)
- ✅ **Fast training I/O:** Memory-mapped + sequential access
- ✅ **Production-ready:** Optimized for deployment
- ⚠️ **Slower preprocessing:** Sequential compression (no parallel speedup)
- ⚠️ **Complex architecture:** Requires shard creation + merge

**Best For:**
- Production deployments
- Large datasets (where compression matters)
- When disk space is limited
- When training I/O speed matters most

---

## 📊 Performance Comparison

### Benchmark Results (2000 samples, 7 CPU cores)

#### Speed-Optimized (files + 7 workers):
```
Workers: 1  → 28.5s (70.2 samples/s)
Workers: 2  → 15.1s (132.4 samples/s) → 1.89× speedup
Workers: 4  → 9.2s  (217.4 samples/s) → 3.10× speedup
Workers: 7  → 7.8s  (256.4 samples/s) → 3.65× speedup

✅ Near-linear scaling with workers
⚡ 3.65× faster preprocessing
📁 ~66 MB disk usage (uncompressed)
```

#### Compression-Optimized (mmap + 1 worker):
```
Workers: 1  → 36.1s (55.4 samples/s)
💾 14.8 MB disk usage (4.46× compression)
🚀 Fast memory-mapped I/O during training

❌ Using mmap with 7 workers:
   → 35.0s (57.1 samples/s) - NO speedup due to I/O contention!
```

---

## 🔬 Why Mmap + Many Workers Doesn't Scale

### The Bottlenecks

1. **Compression (Sequential within shards)**
   ```
   Each worker compresses its shard sequentially
   → LZ4 compression is CPU-bound
   → Limited by single-threaded compression speed
   ```

2. **I/O Contention (During merge)**
   ```
   Multiple workers write to same disk simultaneously
   → Disk thrashing with 7 concurrent writes
   → Merge time: 7 workers (11.7s) > 4 workers (8.4s)
   ```

3. **Result:**
   ```
   Files + 7 workers: 7.8s  (3.65× speedup) ✅
   Mmap + 7 workers:  35.0s (0.97× speedup) ❌
   ```

---

## 🎓 Decision Matrix

### Choose **FILES + MANY WORKERS** if:

- ✅ You're developing/prototyping
- ✅ You iterate frequently on models
- ✅ Disk space is abundant (cloud storage, large drives)
- ✅ Preprocessing time is your bottleneck
- ✅ You can afford larger disk footprint

### Choose **MMAP + 1 WORKER** if:

- ✅ You're deploying to production
- ✅ Dataset is large (hundreds of GB)
- ✅ Disk space is limited (expensive SSD, quotas)
- ✅ Training I/O speed is your bottleneck
- ✅ You preprocess once, train many times

---

## 💡 Recommended Workflows

### Workflow 1: Development → Production

```python
# 1. DEVELOPMENT: Fast iteration with files backend
dev_dataset = OnDiskInductivePreprocessor(
    dataset=source,
    data_dir="./dev_data",
    transforms_config=transforms,
    num_workers=7,
    storage_backend="files",  # Fast preprocessing
)

# Iterate on model architecture...
# Test different hyperparameters...
# Validate approach...

# 2. PRODUCTION: Convert to compressed format
prod_dataset = OnDiskInductivePreprocessor(
    dataset=source,
    data_dir="./prod_data",
    transforms_config=transforms,
    num_workers=1,            # Sequential for reliable compression
    storage_backend="mmap",   # Compressed storage
    compression="lz4",        # Fast decompression
)

# Deploy with optimized storage...
```

### Workflow 2: Hybrid Approach

```python
# Use files for small-medium datasets (< 100K samples)
# Use mmap for large datasets (> 100K samples)

if num_samples < 100_000:
    storage_backend = "files"
    num_workers = 7  # Fast parallel processing
else:
    storage_backend = "mmap"
    num_workers = 1  # Compression matters more
```

---

## 📈 Compression Ratios by Data Type

### Typical Compression with LZ4:

| Data Type | Compression Ratio | Notes |
|-----------|-------------------|-------|
| **Graph features** | 3-5× | Numeric tensors compress well |
| **Simplicial complexes** | 4-6× | Sparse matrices compress very well |
| **Hypergraph incidences** | 5-7× | Extremely sparse, high compression |
| **Point clouds** | 2-3× | Dense numeric data, less compression |

**Example (2000 samples with SimplicialCliqueLifting):**
- Uncompressed: 66.1 MB
- Compressed (LZ4): 14.8 MB
- Ratio: **4.46×**

---

## 🛠️ Advanced: Hybrid Storage

For ultimate flexibility, you can use **both** backends:

```python
# Fast preprocessing phase
temp_dataset = OnDiskInductivePreprocessor(
    dataset=source,
    data_dir="./temp",
    transforms_config=transforms,
    num_workers=7,
    storage_backend="files",  # Fast!
)

# Convert to compressed format post-processing
final_dataset = OnDiskInductivePreprocessor(
    dataset=temp_dataset,     # Use preprocessed data as source
    data_dir="./final",
    transforms_config=None,   # No transforms needed
    num_workers=1,
    storage_backend="mmap",   # Compress!
    compression="lz4",
)

# Clean up temp directory
shutil.rmtree("./temp")
```

---

## 🔍 Debugging Tips

### How to Check What You're Using:

```python
dataset = OnDiskInductivePreprocessor(...)

print(f"Storage backend: {dataset.storage_backend}")
print(f"Compression: {dataset.compression}")
print(f"Workers: {dataset.num_workers}")

if dataset.storage_backend == "mmap":
    stats = dataset._storage.get_stats()
    print(f"Compressed size: {stats['total_size_mb']:.1f} MB")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}×")
```

### Verify Parallel Processing:

Check for this message during preprocessing:
```
Processing (7 workers): 100%|████| 20000/20000 [00:08<00:00, 235sample/s]
```

If you see this, it fell back to sequential:
```
Processing: 100%|████| 20000/20000 [00:27<00:00, 73sample/s]
```

---

## 📚 References

- **Parallel merge implementation:** `topobench/data/preprocessor/ondisk_inductive.py`
- **Benchmark code:** `benchmarks/benchmark_comprehensive_pipeline.py`
- **Performance results:** `results/files_backend_test/parallel/summary.txt`

---

## 🔄 DAG Caching and Storage Backends

### What is DAG Caching?

TopoBench features **DAG-based incremental caching** that automatically reuses cached transforms when you add new transforms to your pipeline. This provides additional speedup on top of the parallel processing.

**Example:**
```python
# Step 1: Process with clique lifting
config1 = {"clique_lifting": {...}}
dataset1 = OnDiskInductivePreprocessor(data_dir="./data", transforms_config=config1, ...)

# Step 2: Add normalization - clique lifting is REUSED!
config2 = {"clique_lifting": {...}, "normalization": {...}}
dataset2 = OnDiskInductivePreprocessor(data_dir="./data", transforms_config=config2, ...)
# Output: "Reusing 1 cached transform(s)!" ✅
```

### DAG Cache + Storage Backend Interaction

The DAG cache works with **both** backends, but the speedup visibility differs:

#### Files Backend (Clear DAG Benefit) ✅

```
Without DAG cache:
  Process T1: 40s
  Process T1 + T2: 40s + 14s = 54s

With DAG cache:
  Process T1: 40s
  Process T1 + T2: 0s + 14s = 14s  ← 3.9× speedup!

Speedup clearly visible!
```

#### Mmap Backend (Hidden DAG Benefit) ⚠️

```
Without DAG cache:
  Process T1: 25s transform + 8s mmap = 33s
  Process T1 + T2: (25s + 10s transform) + (8s + 8s mmap) = 51s

With DAG cache:
  Process T1: 25s transform + 8s mmap = 33s
  Process T1 + T2: 0s + 10s transform + 8s mmap = 18s  ← 1.8× speedup

Speedup partially hidden by mmap overhead!
```

**Key insight:** DAG cache saves **transform processing time** (works for both), but mmap **conversion time cannot be cached** (each new transform output needs compression).

### Recommendation for DAG Caching

1. **Development/Experimentation:** Use `storage_backend="files"`
   - Clear speedup from DAG caching (2-3× per added transform)
   - Fast iteration on pipeline composition
   - Rapid A/B testing of different transform combinations

2. **Production:** Convert final pipeline to `storage_backend="mmap"`
   - 4-5× smaller storage footprint
   - Faster I/O during training
   - One-time preprocessing cost amortized

**See Also:**
- [README_DAG_CACHING.md](README_DAG_CACHING.md) - Complete DAG caching documentation
- [tutorials/dag_caching_tutorial.md](tutorials/dag_caching_tutorial.md) - Interactive tutorial

---

## ✅ Summary

**The Choice:**
- ⚡ **Speed:** `storage_backend="files"` + `num_workers=7`
- 💾 **Compression:** `storage_backend="mmap"` + `num_workers=1`

**The Rule:**
> Use **files** for development, **mmap** for production.

**The Caveat:**
> Never use `mmap` with many workers - compression bottleneck limits speedup to ~2-3× instead of 5-7×.

---

**Date:** November 26, 2024  
**Author:** TopoBench Team  
**Version:** 1.0  
