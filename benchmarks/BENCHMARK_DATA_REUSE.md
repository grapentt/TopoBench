# Benchmark Data Reuse Strategy

## Problem

Currently, each benchmark creates its own preprocessed data:
- `benchmark_parallel_speedup.py` - Creates lifted data with N workers
- `benchmark_lifting_normalization.py` - Creates lifted data  
- `benchmark_dag_recomputation.py` - Creates lifted data multiple times
- `benchmark_training_epochs.py` - Creates lifted data for training
- `benchmark_memory_efficiency.py` - Creates lifted data for memory tests

This means **we're preprocessing the same data multiple times**, wasting time and resources.

## Solution Options

### Option 1: Shared Preprocessing Stage ⭐ **RECOMMENDED**

Create a preprocessing stage that all benchmarks share:

```python
# benchmarks/run_comprehensive_benchmarks.py (modified)

def preprocess_shared_data(config, output_dir):
    """Create shared preprocessed dataset for all benchmarks."""
    shared_dir = output_dir / "_shared_data"
    
    # Create dataset once
    source = SyntheticGraphDataset(...)
    transforms_config = OmegaConf.create({
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,
        },
    })
    
    dataset = OnDiskInductivePreprocessor(
        dataset=source,
        data_dir=shared_dir,
        transforms_config=transforms_config,
        num_workers=4,
        storage_backend="mmap",
    )
    
    return shared_dir

def run_benchmarks(config, output_dir):
    # Preprocess once
    shared_data_dir = preprocess_shared_data(config, output_dir)
    
    # Pass shared_data_dir to each benchmark
    for benchmark in benchmarks:
        run_benchmark(benchmark, output_dir, config, shared_data_dir)
```

**Pros:**
- ✅ Preprocess once, reuse everywhere
- ✅ Demonstrates DAG caching in action!
- ✅ Much faster overall (save 5-10 minutes per run)
- ✅ More realistic (production workflows reuse data)

**Cons:**
- ⚠️ Need to ensure all benchmarks use compatible configs
- ⚠️ Slightly more complex runner logic

### Option 2: Merge Related Benchmarks

Combine benchmarks that naturally share data:

```python
# benchmarks/benchmark_preprocessing_and_training.py (new)

def benchmark_full_pipeline(dataset_size, num_epochs):
    """Benchmark preprocessing + training in one go."""
    
    # 1. Preprocess with parallel workers (benchmark parallel speedup)
    # 2. Measure DAG caching (benchmark recomputation)
    # 3. Train for N epochs (benchmark training)
    # 4. Track memory (benchmark memory efficiency)
    
    return {
        "parallel": parallel_results,
        "dag": dag_results,
        "training": training_results,
        "memory": memory_results,
    }
```

**Pros:**
- ✅ Natural workflow
- ✅ Shares all data automatically
- ✅ Single output with multiple metrics

**Cons:**
- ⚠️ Less modular (can't run individual benchmarks)
- ⚠️ Mixed concerns (preprocessing + training)
- ⚠️ Harder to maintain/extend

### Option 3: Status Quo (Current Approach)

Keep benchmarks separate, each preprocessing their own data.

**Pros:**
- ✅ Simple, modular
- ✅ Each benchmark is independent
- ✅ Easy to run individual benchmarks

**Cons:**
- ❌ Wasteful (reprocesses same data 5+ times)
- ❌ Slow (5-10 extra minutes)
- ❌ Doesn't demonstrate real workflow

## Recommendation: Option 1 (Shared Preprocessing)

Implement shared preprocessing because:

1. **Demonstrates DAG caching in action**: The shared data gets reused by multiple benchmarks, showing real-world efficiency

2. **Faster benchmarks**: Save 5-10 minutes per comprehensive run

3. **More realistic**: In production, you preprocess once and reuse for:
   - Training
   - Validation
   - Hyperparameter tuning
   - Model evaluation

4. **Still modular**: Individual benchmarks can still run independently (they'll just preprocess if shared data doesn't exist)

## Implementation Plan

### Phase 1: Add shared preprocessing to runner

```python
# benchmarks/run_comprehensive_benchmarks.py

def setup_shared_data(config, output_dir):
    """Create shared preprocessed data if needed."""
    shared_dir = output_dir / "_shared_data"
    cache_file = shared_dir / ".cache_info.json"
    
    # Check if cache exists and is valid
    if cache_file.exists():
        with open(cache_file) as f:
            cache_info = json.load(f)
        if cache_info["dataset_size"] == config.dataset_size:
            print(f"✓ Reusing shared data from {shared_dir}")
            return shared_dir
    
    # Preprocess
    print(f"📦 Preprocessing shared data...")
    source = SyntheticGraphDataset(
        num_samples=config.dataset_size,
        num_nodes=50,
        num_features=16,
    )
    
    transforms_config = OmegaConf.create({
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,
        },
    })
    
    dataset = OnDiskInductivePreprocessor(
        dataset=source,
        data_dir=shared_dir,
        transforms_config=transforms_config,
        num_workers=4,
        storage_backend="mmap",
    )
    
    # Save cache info
    shared_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({
            "dataset_size": config.dataset_size,
            "complex_dim": 2,
            "created_at": time.time(),
        }, f)
    
    return shared_dir
```

### Phase 2: Update benchmarks to accept shared data

```python
# benchmarks/benchmark_training_epochs.py

def benchmark_training(
    dataset_size: int,
    shared_data_dir: Path | None = None,  # NEW
    ...
):
    if shared_data_dir and shared_data_dir.exists():
        # Reuse preprocessed data!
        print(f"✓ Using shared preprocessed data from {shared_data_dir}")
        dataset = OnDiskInductivePreprocessor(
            dataset=...,
            data_dir=shared_data_dir,
            transforms_config=...,
            force_reload=False,  # Use cache!
            storage_backend="mmap",
        )
    else:
        # Preprocess locally
        dataset = OnDiskInductivePreprocessor(...)
```

### Phase 3: Update CLI to support shared data

```bash
# Run with shared preprocessing (default)
python benchmarks/run_comprehensive_benchmarks.py \\
    --config benchmarks/configs/comprehensive.yaml \\
    --output results/comprehensive

# Run without shared preprocessing (independent)
python benchmarks/run_comprehensive_benchmarks.py \\
    --config benchmarks/configs/comprehensive.yaml \\
    --output results/comprehensive \\
    --no-shared-data
```

## Benefits Summary

**Time Savings Example** (for comprehensive config):

Without shared data:
- parallel_speedup: 250s (preprocessing)
- lifting_normalization: 45s (preprocessing)
- dag_recomputation: 120s (preprocessing 4×)
- training: 180s (preprocessing + training)
- memory: 90s (preprocessing 3×)
- **Total: ~685s (~11 min)**

With shared data:
- **Shared preprocessing: 60s (once!)**
- parallel_speedup: 5s (just speedup measurement)
- lifting_normalization: 5s (just timing)
- dag_recomputation: 70s (cache validation)
- training: 120s (just training)
- memory: 30s (just memory measurement)
- **Total: ~290s (~5 min)**

**Savings: ~400s (6-7 minutes) per comprehensive run!** 🚀

## Next Steps

1. ✅ Implement shared preprocessing in `run_comprehensive_benchmarks.py`
2. ✅ Update each benchmark to accept `shared_data_dir` parameter
3. ✅ Add `--no-shared-data` CLI flag for independent runs
4. ✅ Update documentation to highlight data reuse efficiency
5. ✅ Add test to verify shared data works correctly
