# TopoBench Benchmarks - Quick Start Guide

## 🚀 Running Benchmarks

### Option 1: Use Comprehensive Pipeline (Recommended)

The comprehensive pipeline runs multiple benchmarks in one command:

```bash
# Run all benchmarks with test configuration
.venv/bin/python benchmarks/benchmark_comprehensive_pipeline.py \
    --config benchmarks/configs/test.yaml \
    --output results/test_run \
    --benchmarks all

# Run specific benchmarks
.venv/bin/python benchmarks/benchmark_comprehensive_pipeline.py \
    --config benchmarks/configs/test.yaml \
    --output results/memory_only \
    --benchmarks memory

# Available benchmarks: speedup, dag, memory, training
# Separate multiple with commas: --benchmarks speedup,memory
```

### Option 2: Run Individual Benchmarks

```bash
# Memory efficiency (preprocessing only - fast!)
.venv/bin/python benchmarks/benchmark_memory_efficiency.py \
    --output results/memory

# Parallel speedup
.venv/bin/python benchmarks/benchmark_parallel_speedup.py \
    --output results/parallel

# DAG cache reuse
.venv/bin/python benchmarks/benchmark_dag_recomputation.py \
    --output results/dag
```

---

## 📊 Custom Dataset Sizes

### Method 1: Create Custom Config File

Create `benchmarks/configs/custom.yaml`:

```yaml
# Parallel speedup benchmark
parallel:
  dataset_size: 1000
  runs: 3

# DAG cache benchmark  
dag_cache:
  dataset_size: 500
  runs: 3

# Memory efficiency benchmark (preprocessing only)
memory:
  dataset_sizes: [100, 500, 1000, 2000]  # Multiple sizes
  runs: 1

# Training memory benchmark (optional, very slow!)
training:
  dataset_sizes: [5, 10, 20]  # Keep small!
  runs: 1
```

Then run:
```bash
.venv/bin/python benchmarks/benchmark_comprehensive_pipeline.py \
    --config benchmarks/configs/custom.yaml \
    --output results/custom_run \
    --benchmarks all
```

### Method 2: Modify Existing Config

Edit `benchmarks/configs/test.yaml` directly:

```yaml
memory:
  dataset_sizes: [200, 500, 1000]  # ← Change these numbers
  runs: 1
```

### Method 3: Command-Line Arguments (Individual Benchmarks)

For standalone memory benchmark:
```bash
# Edit the script temporarily or use Python
.venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from benchmarks.benchmark_memory_efficiency import main
import argparse

# Override config
config = {
    'dataset_sizes': [100, 300, 500, 1000, 2000],  # Your custom sizes
}

# Run benchmark
main(config=config, output_dir='results/custom_memory')
"
```

---

## ⚙️ Available Configurations

| Config File | Purpose | Dataset Sizes | Speed |
|-------------|---------|---------------|-------|
| `test.yaml` | Quick testing | Small (50-200) | Fast (~1 min) |
| `production.yaml` | Production run | Medium (500-2000) | Medium (~5 min) |
| `publication.yaml` | Paper results | Large (1000-10000) | Slow (~30 min) |

---

## 📈 Understanding Results

### Memory Benchmark Output

```json
{
  "preprocessing_memory": {
    "inmemory": [
      {"dataset_size": 50, "peak_memory_mb": 873, "delta_memory_mb": 11}
    ],
    "ondisk": [
      {"dataset_size": 50, "peak_memory_mb": 866, "delta_memory_mb": 4}
    ]
  }
}
```

**Key Metrics:**
- `peak_memory_mb`: Total process memory at peak
- `delta_memory_mb`: Memory **growth** from baseline (this is the important metric!)
- **In-memory delta grows** with dataset size (linear)
- **On-disk delta stays constant** (~4MB regardless of size)

### Parallel Speedup Output

```json
{
  "parallel_speedup": {
    "1": {"time_seconds": 10.5},
    "2": {"time_seconds": 5.8},
    "4": {"time_seconds": 3.2},
    "None": {"time_seconds": 2.1}
  }
}
```

Shows preprocessing time with different worker counts.

---

## 🎯 Recommended Workflow

### For Quick Testing (< 2 minutes)
```bash
.venv/bin/python benchmarks/benchmark_comprehensive_pipeline.py \
    --config benchmarks/configs/test.yaml \
    --output results/quick_test \
    --benchmarks memory,speedup
```

### For Production Results (~ 10 minutes)
```bash
.venv/bin/python benchmarks/benchmark_comprehensive_pipeline.py \
    --config benchmarks/configs/production.yaml \
    --output results/production_run \
    --benchmarks all
```

### For Publication (~ 30+ minutes)
```bash
.venv/bin/python benchmarks/benchmark_comprehensive_pipeline.py \
    --config benchmarks/configs/publication.yaml \
    --output results/paper_results \
    --benchmarks speedup,dag,memory
    # Note: Skip 'training' for publication - too slow
```

---

## ⚠️ Important Notes

1. **Memory Benchmark** measures preprocessing only (fast, safe)
2. **Training Benchmark** includes full model training (very slow, keep dataset_sizes < 20)
3. **Always use `.venv/bin/python`** to ensure correct environment
4. **Results are deterministic** with same seed and config
5. **Node count = 5** by default for fast clique lifting
6. **Multiprocessing isolation** prevents memory contamination between runs

---

## 🐛 Troubleshooting

**Benchmark hangs:**
- Check if SyntheticGraphDataset has the bounds check fix (lines 337-339 in `benchmarks/utils.py`)

**OOM crashes:**
- Reduce `dataset_sizes` in config
- Run only `memory` benchmark (not `training`)
- Close other applications

**Slow performance:**
- Check node count (should be 5 for quick benchmarks)
- Reduce dataset sizes
- Use fewer `runs`

---

## 📁 Output Structure

```
results/
└── your_run_name/
    ├── comprehensive_results.json    # All results
    ├── memory_plot.png               # Visualization (if generated)
    └── system_info.json              # System specs
```

---

## ✅ Quick Validation

Test everything works:
```bash
# Should complete in < 1 minute
.venv/bin/python benchmarks/benchmark_comprehensive_pipeline.py \
    --config benchmarks/configs/test.yaml \
    --output results/validation \
    --benchmarks memory

# Check results
cat results/validation/comprehensive_results.json | head -20
```

Expected: See `preprocessing_memory` with results for 50, 100, 200 samples.
