# On-Disk Preprocessing: Performance Guide

## ⚡ vs 💾: Speed vs Compression Trade-off

When using `OnDiskInductivePreprocessor`, choose your strategy:

### For Development (Speed First)
```python
dataset = OnDiskInductivePreprocessor(
    dataset=source,
    data_dir="./data",
    transforms_config=transforms,
    num_workers=7,              # Use all cores
    storage_backend="files",    # No compression
)
# ⚡ 3-4× faster preprocessing
# 📁 ~4-5× larger disk usage
```

### For Production (Compression First)
```python
dataset = OnDiskInductivePreprocessor(
    dataset=source,
    data_dir="./data",
    transforms_config=transforms,
    num_workers=1,              # Sequential
    storage_backend="mmap",     # Compressed
    compression="lz4",
)
# 💾 4-5× smaller disk footprint
# 🚀 Faster I/O during training
# ⏱️  Slower preprocessing
```

**Performance (2000 samples, 7 cores):**
- Files + 7 workers: 8.9s (3.38× speedup) ⚡
- Mmap + 1 worker: 36.0s (4.46× compression) 💾

**Rule of thumb:** Use **files** for dev, **mmap** for production.

> ⚠️ **Important:** Don't use `mmap` with many workers - compression creates a bottleneck that limits speedup to ~2× instead of 3-4×.

See [SPEED_VS_COMPRESSION_TRADEOFF.md](SPEED_VS_COMPRESSION_TRADEOFF.md) for detailed analysis.
