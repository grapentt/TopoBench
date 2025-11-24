# B1 Benchmark Engineering Plan 🔬

**For**: World-Class Benchmark Engineer  
**Mission**: Create comprehensive benchmarks to prove TopoBench's claims  
**Deadline**: 4-5 hours  
**Priority**: CRITICAL

---

## 🎯 Executive Summary

**Problem**: Competitor (PR #213 https://github.com/geometric-intelligence/TopoBench/pull/213/files) has evidence-based claims. We have claims without proof!

**What They Have**:
- Memory profiling utilities
- Evidence of memory savings
- Tested with multiple models

**What We Need to Prove**:
1. **Memory Efficiency**: O(1) vs O(N) memory usage (15-20× savings)
2. **Augmentation Speedup**: 10-100× faster with two-tier transforms
3. **Performance Stack**: Combined effect of all optimizations

---

## 📋 Three Critical Benchmarks

### 1. Memory Profiling (2-3 hours) 🔥 HIGHEST PRIORITY

**File**: `topobench/benchmarks/memory_profiling.py`

**Goal**: Prove O(1) constant memory vs O(N) in-memory

**Key Measurements**:
- Test dataset sizes: [100, 500, 1000, 2000, 5000] samples
- InMemory: Peak memory, RSS growth, memory per sample
- OnDisk: Peak memory, RSS growth (should be constant)
- Tools: `tracemalloc`, `psutil.Process().memory_info()`

**Expected Results**:
- InMemory @ 5000 samples: ~450 MB peak (O(N))
- OnDisk @ 5000 samples: ~20 MB peak (O(1))
- **Savings: 15-25× less memory**

**Deliverables**:
1. Standalone script with MemoryProfiler class
2. JSON results: `results/memory_benchmark.json`
3. Plot: `results/memory_comparison.png` (4 subplots)
4. Summary table for README

**Implementation Notes**:
- Compare `GeneratedInductiveDataset` (in-memory) vs `OnDiskInductivePreprocessor`
- Disable cache during measurement (`cache_size=0`)
- Force GC between measurements
- Access 100 samples to trigger memory usage

---

### 2. Augmentation Speedup (2 hours) 🔥 HIGHEST PRIORITY

**File**: `topobench/benchmarks/augmentation_speedup.py`

**Goal**: Prove 10-100× speedup for N augmentation experiments

**Scenario**: Researcher tests 10 different noise levels (0.15, 0.30, ..., 1.50)

**Traditional Approach**:
- Each parameter requires FULL preprocessing
- Time: N × (preprocessing_time)
- Example: 10 experiments × 30s = 300s

**Two-Tier Approach**:
- Heavy transforms cached once
- Light transforms changed at runtime
- Time: 1 × preprocessing_time + N × 0s
- Example: 30s + 0s = 30s
- **Speedup: 10×**

**Key Code**:
```python
# Traditional (baseline)
for noise in [0.15, 0.30, 0.45, ...]:
    config = {"lifting": {...}, "augmentation": {"noise_std": noise}}
    prep = OnDiskInductivePreprocessor(..., transform_tier="all_heavy")
    # Full preprocessing each time!

# Two-Tier (our approach)
prep = OnDiskInductivePreprocessor(..., transform_tier="auto")  # Once
for noise in [0.15, 0.30, 0.45, ...]:
    prep.transform_pipeline.light_transforms[0].noise_std = noise
    # INSTANT - heavy transforms cached!
```

**Deliverables**:
1. Standalone script with AugmentationBenchmark class
2. JSON results with timing breakdown
3. Plot showing speedup
4. README table

---

### 3. Comprehensive Performance Stack (1 hour) ⚡ MEDIUM PRIORITY

**File**: `topobench/benchmarks/full_stack_benchmark.py`

**Goal**: Show combined effect of all optimizations

**What to Measure**:
1. Baseline (no optimizations): Sequential, file-based, no cache
2. With parallel: 4-8× speedup
3. With mmap: +2-3× I/O speedup
4. With compression: 1.5-2× disk savings
5. With LRU cache: 1.2-1.3× training speedup
6. **Combined**: Compound effect

**Deliverables**:
1. Single script measuring incremental improvements
2. Waterfall chart showing cumulative speedup
3. README performance table

---

## 📦 Deliverable Structure

```
topobench/
  benchmarks/
    __init__.py
    memory_profiling.py       # Benchmark 1
    augmentation_speedup.py   # Benchmark 2
    full_stack_benchmark.py   # Benchmark 3
    
results/
  memory_benchmark.json
  memory_comparison.png
  augmentation_benchmark.json
  augmentation_speedup.png
  full_stack_results.json
  full_stack_comparison.png
  
README.md                     # Updated with results
```

---

## 🎯 Success Criteria

### Memory Profiling ✅
- Prove O(1) constant memory
- Show 15-25× memory savings vs in-memory
- Generate publication-quality plots

### Augmentation Speedup ✅
- Prove 10-100× speedup claim
- Show cache reuse working
- Demonstrate practical benefit

### Full Stack ✅
- Show compound effect of optimizations
- Provide complete performance story
- Enable impressive README claims

---

## 🚀 Quick Start for Engineer

```bash
# 1. Setup
cd /home/tgrapentin/personal/tdl/Topo2/TopoBench
source .venv/bin/activate
mkdir -p topobench/benchmarks results

# 2. Run existing tests to understand codebase
pytest test/data/preprocessor/test_ondisk_inductive.py -v

# 3. Review existing benchmark
python benchmark_storage_approaches.py --help

# 4. Implement three benchmarks
#    - Use existing code as reference
#    - Follow structure from benchmark_storage_approaches.py
#    - Use psutil, tracemalloc, matplotlib

# 5. Run benchmarks
python topobench/benchmarks/memory_profiling.py
python topobench/benchmarks/augmentation_speedup.py
python topobench/benchmarks/full_stack_benchmark.py

# 6. Review results
ls -lh results/
```

---

## 📊 Expected README Update

```markdown
## 🚀 Performance Benchmarks

**Hardware**: [Your specs here]

### Memory Efficiency
| Dataset Size | InMemory | OnDisk | Savings |
|--------------|----------|--------|---------|
| 1,000 samples | 98 MB | 12 MB | **8×** less |
| 5,000 samples | 456 MB | 18 MB | **25×** less |

### Augmentation Speedup
| Experiments | Traditional | Two-Tier | Speedup |
|-------------|-------------|----------|---------|
| 10 noise levels | 300s | 30s | **10×** faster |
| 50 noise levels | 1500s | 30s | **50×** faster |

### Combined Performance (values are probably too optimistic)
- ✅ **4-8× preprocessing speedup** (parallel processing)
- ✅ **2-3× I/O speedup** (memory-mapped storage)
- ✅ **10-100× augmentation speedup** (two-tier transforms)
- ✅ **15-25× memory savings** (O(1) vs O(N))
- ✅ **1.5-2× disk savings** (compression)
```
---

## ✅ Validation Checklist

Before declaring success:

- [ ] Memory profiling shows clear O(1) vs O(N) scaling
- [ ] Augmentation speedup shows 10-100× improvement
- [ ] All plots are publication-quality (300 DPI)
- [ ] JSON results are properly formatted
- [ ] README updated with actual numbers
- [ ] Scripts run successfully end-to-end
- [ ] Results are reproducible

---

**Good luck! You're creating the evidence that will win B1! 🏆**
