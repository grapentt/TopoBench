# TopoBench B1 Benchmarking Suite

**Purpose**: Rigorous, academically sound benchmarks proving TopoBench's B1 implementation performance claims.

**Philosophy**: Intellectual honesty and reproducibility above all else.

## Benchmark Categories

### 1. Memory Profiling (`memory_profiling.py`)
**Claim**: O(1) constant memory vs O(N) in-memory approach  
**Evidence**: Peak memory usage across dataset sizes [100, 500, 1K, 5K, 10K]  
**Metrics**: Peak RSS, memory per sample, scaling behavior

### 2. Parallel Processing (`parallel_speedup.py`)
**Claim**: 4-8× speedup with parallel preprocessing  
**Evidence**: Processing time with [1, 2, 4, 8] workers  
**Metrics**: Throughput (samples/sec), speedup vs baseline, efficiency

### 3. Storage Backend (`storage_comparison.py`)
**Claim**: 2-3× faster I/O with memory-mapped storage  
**Evidence**: Read/write performance, file vs mmap  
**Metrics**: Read latency, write throughput, random access performance

### 4. Compression (`compression_comparison.py`)
**Claim**: 1.3-1.6× disk savings, fast decompression  
**Evidence**: Compression ratio, read speed for [none, lz4, zstd]  
**Metrics**: Disk usage, compression time, decompression speed

### 5. Cache Performance (`cache_benchmark.py`)
**Claim**: 1.2-1.3× training speedup with LRU cache  
**Evidence**: Training epoch time with/without cache  
**Metrics**: Cache hit rate, epoch time, speedup

### 6. End-to-End Workflow (`e2e_benchmark.py`)
**Claim**: 12-18× faster complete workflow  
**Evidence**: Full preprocessing + training pipeline  
**Metrics**: Total time, memory peak, disk usage

## Running Benchmarks

```bash
# Activate environment
source .venv/bin/activate

# Run all benchmarks (takes ~30-60 minutes)
python benchmarks/run_all.py

# Run individual benchmarks
python benchmarks/memory_profiling.py
python benchmarks/parallel_speedup.py
python benchmarks/storage_comparison.py
python benchmarks/compression_comparison.py
python benchmarks/cache_benchmark.py
python benchmarks/e2e_benchmark.py
```

## Output Structure

```
results/
├── memory/
│   ├── raw_data.json
│   ├── memory_scaling.png
│   └── summary.txt
├── parallel/
│   ├── raw_data.json
│   ├── speedup_curves.png
│   └── summary.txt
├── storage/
│   ├── raw_data.json
│   ├── io_comparison.png
│   └── summary.txt
├── compression/
│   ├── raw_data.json
│   ├── compression_tradeoffs.png
│   └── summary.txt
├── cache/
│   ├── raw_data.json
│   ├── hit_rate_analysis.png
│   └── summary.txt
├── e2e/
│   ├── raw_data.json
│   ├── workflow_comparison.png
│   └── summary.txt
└── BENCHMARK_REPORT.md  # Comprehensive report
```

## Reproducibility

- All benchmarks use fixed random seeds
- System info captured (CPU, RAM, disk)
- Multiple runs with statistical analysis (mean, std, confidence intervals)
- Raw data saved in JSON for independent analysis
- Code is self-contained and documented

## Academic Standards

✅ **Reproducibility**: Fixed seeds, documented environment  
✅ **Statistical rigor**: Multiple runs, confidence intervals  
✅ **Honest reporting**: Show failures, limitations, variance  
✅ **Clear methodology**: Documented assumptions, parameters  
✅ **Publication quality**: High-DPI plots, clear labels, legends
