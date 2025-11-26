# Memory Optimization Guide for OGBG-molpcba and Large Datasets

## 🔍 Memory Issues with OnDisk Datasets

Despite using OnDiskInductivePreprocessor for O(1) memory usage, you may still experience OOM errors with large datasets. This guide explains **why** and **how to fix it**.

---

## 🐛 Root Causes of Memory Leaks

### 1. **In-Memory Caching** (PRIMARY CAUSE)
**Problem:** Default `cache_size=100` caches 100 samples in RAM
- Each molecular graph after lifting: 100KB - 1MB
- With 100 samples: 10MB - 100MB
- **With DataLoader workers:** `num_workers × cache_size` samples cached!
  - 7 workers × 100 samples = **700 samples = 70MB - 700MB**

**Solution:** Disable caching for large datasets
```yaml
preprocessor:
  cache_size: 0  # CRITICAL: Set to 0 for large datasets
```

### 2. **DataLoader Worker Memory Leaks**
**Problem:** PyTorch DataLoader workers can accumulate memory
- Each worker creates a copy of the dataset object
- Workers aren't properly garbage collected
- PyG Data objects have complex tensor structures that leak

**Solution:** Use single-threaded loading
```yaml
dataset:
  dataloader_params:
    num_workers: 0  # Single-threaded to avoid worker memory leaks
    persistent_workers: false  # Don't keep workers alive
    pin_memory: false  # Disable to save memory
```

### 3. **Gradient Accumulation**
**Problem:** PyTorch Lightning may accumulate gradients/metrics
- Metrics stored for all batches
- Gradients not cleared properly
- Progress bars keep batch outputs

**Solution:** Configure trainer for memory efficiency
```yaml
trainer:
  gradient_clip_val: 1.0  # Prevent gradient explosion
  log_every_n_steps: 100  # Reduce logging overhead
  enable_model_summary: false  # Reduce memory from summary
  limit_val_batches: 50  # Limit validation batches
```

### 4. **PyG Data Object Accumulation**
**Problem:** PyTorch Geometric Data objects can accumulate
- Complex nested tensor structures
- Sparse matrices not released
- Transform caching intermediate results

**Solution:** Use explicit garbage collection (see script below)

---

## ✅ Complete Memory-Efficient Configuration

### Experiment Config (`ogbg_molpcba_scn2_full.yaml`)
```yaml
# Dataset
dataset:
  loader:
    parameters:
      subset_size: null  # Full dataset
  
  dataloader_params:
    batch_size: 32  # Reduce if still OOM (try 16, 8, 4)
    num_workers: 0  # CRITICAL: 0 workers
    persistent_workers: false
    pin_memory: false

# Preprocessor
preprocessor:
  mode: ondisk
  cache_size: 0  # CRITICAL: Disable cache
  storage_backend: mmap  # Compressed, memory-mapped
  
# Trainer
trainer:
  max_epochs: 50
  gradient_clip_val: 1.0
  log_every_n_steps: 100
  enable_model_summary: false
  limit_val_batches: 50
```

---

## 🚀 Usage Examples

### Quick Test (1K samples)
```bash
# Clear old cache
rm -rf datasets/ogbg_molpcba/preprocessed

# Test with subset
python -m topobench.run \
    experiment=ogbg_molpcba_scn2_full \
    dataset.loader.parameters.subset_size=1000 \
    preprocessor.cache_size=0 \
    dataset.dataloader_params.num_workers=0
```

### Medium Scale (10K samples)
```bash
rm -rf datasets/ogbg_molpcba/preprocessed

python -m topobench.run \
    experiment=ogbg_molpcba_scn2_full \
    dataset.loader.parameters.subset_size=10000 \
    preprocessor.cache_size=0
```

### Full Dataset (437K samples) - Memory-Efficient Mode
```bash
rm -rf datasets/ogbg_molpcba/preprocessed

python train_ogbg_memory_efficient.py \
    experiment=ogbg_molpcba_scn2_full \
    dataset.loader.parameters.subset_size=null \
    preprocessor.cache_size=0
```

The memory-efficient script adds:
- Explicit garbage collection after each batch
- Cache clearing after each epoch
- Memory monitoring and logging
- Aggressive cleanup

---

## 📊 Expected Memory Usage

| Configuration | Preprocessing | Training | Total | Notes |
|--------------|---------------|----------|-------|-------|
| **Subset 1K** | ~50MB | ~500MB | ~550MB | Good for testing |
| **Subset 10K** | ~100MB | ~1.5GB | ~1.6GB | Medium scale |
| **Full 437K** | ~200MB | ~2-3GB | ~2.5GB | With optimizations |
| **Full + cache=100** | ~200MB | ~8-12GB | **~12GB** | OOM likely! |
| **Full + workers=7** | ~200MB | ~15-20GB | **~20GB** | OOM certain! |

**Key Insight:** With proper optimizations, full dataset should fit in **4GB RAM**.

---

## 🛠️ Troubleshooting OOM

### If you still get OOM after optimizations:

#### 1. Reduce Batch Size
```bash
python -m topobench.run \
    experiment=ogbg_molpcba_scn2_full \
    dataset.dataloader_params.batch_size=16  # or 8, or 4
```

#### 2. Use Gradient Accumulation
Train with smaller batches but accumulate to simulate larger batches:
```yaml
trainer:
  accumulate_grad_batches: 4  # Simulate batch_size * 4
```

#### 3. Limit Training Data
```bash
python -m topobench.run \
    experiment=ogbg_molpcba_scn2_full \
    dataset.loader.parameters.subset_size=50000  # Use 50K instead of 437K
```

#### 4. Use GPU (if available)
GPU has separate memory and can offload from RAM:
```bash
python -m topobench.run \
    experiment=ogbg_molpcba_scn2_full \
    trainer.accelerator=gpu
```

#### 5. Monitor Memory During Training
```bash
# Install psutil
pip install psutil

# Run memory-efficient script
python train_ogbg_memory_efficient.py experiment=ogbg_molpcba_scn2_full
```

This will print memory usage every 100 batches:
```
[Memory] After 100 batches: 2453.2 MB
[Memory] After 200 batches: 2461.8 MB  # Should be stable!
[Memory] After 300 batches: 2459.3 MB
```

If memory keeps growing (e.g., +10MB per batch), there's a leak.

---

## 🔧 Advanced: Finding Memory Leaks

### Check if DataLoader is leaking
```python
import gc
import torch

# Force garbage collection
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Check tensor count
print(f"Tensors in memory: {len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])}")
```

### Profile memory usage
```python
import tracemalloc

tracemalloc.start()

# ... train for one epoch ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

---

## 📝 Configuration Checklist

Before training on full dataset, verify:

- [ ] `preprocessor.cache_size: 0`
- [ ] `dataset.dataloader_params.num_workers: 0`
- [ ] `dataset.dataloader_params.persistent_workers: false`
- [ ] `dataset.dataloader_params.pin_memory: false`
- [ ] `trainer.enable_model_summary: false`
- [ ] Batch size reasonable (start with 32, reduce if OOM)
- [ ] Old preprocessed cache cleared (`rm -rf datasets/.../preprocessed`)

---

## 🎯 Recommended Workflow

### For Testing (fast iterations)
```bash
python -m topobench.run \
    experiment=ogbg_molpcba_scn2_full \
    dataset.loader.parameters.subset_size=1000 \
    trainer.max_epochs=5
```

### For Validation (check performance)
```bash
python -m topobench.run \
    experiment=ogbg_molpcba_scn2_full \
    dataset.loader.parameters.subset_size=10000 \
    trainer.max_epochs=20
```

### For Full Training (production)
```bash
# Clear cache first!
rm -rf datasets/ogbg_molpcba/preprocessed

# Use memory-efficient script
python train_ogbg_memory_efficient.py \
    experiment=ogbg_molpcba_scn2_full \
    trainer.max_epochs=50 \
    logger=wandb  # or csv
```

**Expected time:**
- Preprocessing: 2-3 hours (one-time)
- Training/epoch: 20-30 minutes
- Total (50 epochs): ~25-30 hours

---

## 🏆 Summary

**Memory issues with OnDisk datasets are caused by:**
1. **In-memory caching** (cache_size > 0)
2. **DataLoader workers** (num_workers > 0)
3. **PyTorch/Lightning overhead**

**Solution: Disable all memory-consuming features**
- ✅ cache_size=0
- ✅ num_workers=0
- ✅ Use memory-efficient trainer settings
- ✅ Use explicit garbage collection
- ✅ Monitor memory during training

With these optimizations, **full OGBG-molpcba (437K samples) should train in 2.5-3GB RAM**!

---

## 📚 Related Files

- `configs/experiment/ogbg_molpcba_scn2_full.yaml` - Full training config (optimized)
- `train_ogbg_memory_efficient.py` - Memory-efficient training script
- `topobench/data/preprocessor/ondisk_inductive.py` - OnDisk preprocessor implementation
- `COMPLETE_VICTORY_REPORT.md` - Complete integration status

---

*If you still experience OOM after all optimizations, your system may not have enough RAM for this dataset. Consider using a cloud instance with more memory or further reducing subset_size.*
