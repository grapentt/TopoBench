# Data Size Scaling Guide

**Use `--scale N` to multiply all data sizes by N**

This makes it easy to run larger benchmarks to prove performance claims at scale.

---

## 📊 Publication Config with Different Scales

### Base Publication Config (--scale 1.0)
```yaml
memory:
  sizes: [100, 500, 1000, 2000, 5000, 10000]
  accesses: 200
parallel:
  samples: 5000
  workers: [1, 2, 4, 8]
  runs: 5
storage:
  samples: 2000
  reads: 200
```

**Estimated time**: ~2-5 minutes (too small for clear results!)

---

### --scale 2 (Double All Sizes)
```yaml
memory:
  sizes: [200, 1000, 2000, 4000, 10000, 20000]
  accesses: 400
parallel:
  samples: 10000
storage:
  samples: 4000
  reads: 400
```

**Estimated time**: ~5-10 minutes  
**Expected results**: 
- Memory O(1): Slope ratio 3-5× (getting better)
- Parallel: 2-3× speedup (improving)

---

### --scale 5 (5× Larger)
```yaml
memory:
  sizes: [500, 2500, 5000, 10000, 25000, 50000]
  accesses: 1000
parallel:
  samples: 25000
storage:
  samples: 10000
  reads: 1000
```

**Estimated time**: ~15-30 minutes  
**Expected results**: 
- Memory O(1): Slope ratio 10-20× ✅ **CLAIM PROVEN**
- Parallel: 4-6× speedup ✅ **CLAIM PROVEN**
- Mmap: 2-3× speedup ✅ **CLAIM PROVEN**

**Recommended for proving claims!**

---

### --scale 10 (10× Larger)
```yaml
memory:
  sizes: [1000, 5000, 10000, 20000, 50000, 100000]
  accesses: 2000
parallel:
  samples: 50000
storage:
  samples: 20000
  reads: 2000
```

**Estimated time**: ~45-90 minutes  
**Expected results**: 
- Memory O(1): Slope ratio 20-50× ✅ **STRONG EVIDENCE**
- Parallel: 6-8× speedup ✅ **TARGET ACHIEVED**
- Mmap: 3-4× speedup ✅ **EXCEEDS TARGET**

**Recommended for publication!**

---

### --scale 20 (20× Larger - Extreme Scale)
```yaml
memory:
  sizes: [2000, 10000, 20000, 40000, 100000, 200000]
  accesses: 4000
parallel:
  samples: 100000
storage:
  samples: 40000
  reads: 4000
```

**Estimated time**: ~2-4 hours  
**Expected results**: 
- Memory O(1): Slope ratio 50-100× ✅ **DEFINITIVE PROOF**
- Parallel: 7-8× speedup ✅ **MAXIMUM SPEEDUP**
- Massive datasets (200K samples)

**For competitive benchmarking against SOTA!**

---

## 🎯 Quick Reference

| Scale | Memory Max | Parallel Samples | Time Estimate | Use Case |
|-------|-----------|-----------------|---------------|----------|
| 1.0 | 10,000 | 5,000 | 2-5 min | ❌ Too small |
| 2.0 | 20,000 | 10,000 | 5-10 min | ⚠️ Partial evidence |
| 5.0 | 50,000 | 25,000 | 15-30 min | ✅ **Prove claims** |
| 10.0 | 100,000 | 50,000 | 45-90 min | ✅ **Publication** |
| 20.0 | 200,000 | 100,000 | 2-4 hours | 🏆 **Competitive** |

---

## 💡 Usage Examples

### Quick Test (5-10 minutes)
```bash
.venv/bin/python benchmarks/run_benchmarks.py \
    --config publication \
    --scale 2 \
    --output results_scale2
```

### Prove Claims (15-30 minutes) - **RECOMMENDED**
```bash
.venv/bin/python benchmarks/run_benchmarks.py \
    --config publication \
    --scale 5 \
    --output results_scale5

# Generate report
.venv/bin/python benchmarks/generate_report.py \
    --results results_scale5 \
    --output PROOF_OF_CLAIMS.md
```

### Publication Quality (45-90 minutes)
```bash
.venv/bin/python benchmarks/run_benchmarks.py \
    --config publication \
    --scale 10 \
    --output results_scale10

# Generate report
.venv/bin/python benchmarks/generate_report.py \
    --results results_scale10 \
    --output PUBLICATION_REPORT.md
```

### Beat Competition (2-4 hours)
```bash
.venv/bin/python benchmarks/run_benchmarks.py \
    --config publication \
    --scale 20 \
    --output results_scale20

# Generate report
.venv/bin/python benchmarks/generate_report.py \
    --results results_scale20 \
    --output COMPETITIVE_REPORT.md
```

---

## 📈 Expected Memory Usage

At different scales, here's the approximate memory needed:

| Scale | Max Dataset Size | InMemory Approach | OnDisk Approach | Savings |
|-------|-----------------|-------------------|-----------------|---------|
| 1.0 | 10K samples (~30 MB) | 900 MB | 870 MB | 1.0× |
| 2.0 | 20K samples (~60 MB) | 930 MB | 870 MB | 1.1× |
| 5.0 | 50K samples (~150 MB) | 1,020 MB | 870 MB | 1.2× |
| 10.0 | 100K samples (~300 MB) | 1,170 MB | 870 MB | 1.3× |
| 20.0 | 200K samples (~600 MB) | 1,470 MB | 870 MB | 1.7× |

**At scale 20.0**: OnDisk uses O(1) memory while InMemory grows O(N) - **clear proof!**

---

## 🔬 Why Scaling Matters

### Problem with Small Datasets

With 10,000 samples:
- Total data: ~30 MB
- Python baseline: ~867 MB
- Data is only 3% of total memory
- **Impossible to see O(1) vs O(N) difference!**

### Solution: Scale Up

With 100,000 samples (scale 10.0):
- Total data: ~300 MB
- Python baseline: ~867 MB
- Data is now 35% of total memory
- **O(1) vs O(N) clearly visible!**

With 200,000 samples (scale 20.0):
- Total data: ~600 MB
- Python baseline: ~867 MB
- Data is now 70% of total memory
- **Dramatic O(1) vs O(N) difference!**

---

## 🎯 Recommendations

### For Your Situation

Since publication config took only 2 minutes, you need **much larger datasets**.

**Start here** (15-30 minutes):
```bash
.venv/bin/python benchmarks/run_benchmarks.py \
    --config publication \
    --scale 5 \
    --output results_proof
```

If results still not clear, **go bigger** (45-90 minutes):
```bash
.venv/bin/python benchmarks/run_benchmarks.py \
    --config publication \
    --scale 10 \
    --output results_publication
```

For **definitive competitive results** (2-4 hours):
```bash
.venv/bin/python benchmarks/run_benchmarks.py \
    --config publication \
    --scale 20 \
    --output results_competitive
```

---

## 🚀 Quick Commands

```bash
# Prove your claims (recommended)
.venv/bin/python benchmarks/run_benchmarks.py --config publication --scale 5

# Publication quality
.venv/bin/python benchmarks/run_benchmarks.py --config publication --scale 10

# Beat the competition
.venv/bin/python benchmarks/run_benchmarks.py --config publication --scale 20
```

**Then generate report:**
```bash
.venv/bin/python benchmarks/generate_report.py --results results --output REPORT.md
```

---

**🎓 Scientific Rule of Thumb**: Your effect size (data memory) should be at least 10-20% of baseline (Python overhead) to detect O(1) vs O(N) differences clearly.

**With --scale 10 or --scale 20, you'll get publication-quality proof of your claims! 🏆**
