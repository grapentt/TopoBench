# 🎯 INTEGRATION_PLAN: Architectural Master Plan

**Version:** 1.0 | **Created:** Nov 21, 2025, 00:51 UTC+01:00  
**Status:** ACTIVE | **Priority:** CRITICAL

---

## 🌟 EXECUTIVE SUMMARY

**Mission:** Transform TopoBench's OnDisk infrastructure into a bulletproof system that demonstrably enables topological deep learning at scales where traditional approaches fail.

**Core Philosophy:**
1. **Functionality First, Speed Second** - Prove survival before optimizing
2. **Scientific Rigor** - Every claim backed by reproducible evidence  
3. **Elegant Integration** - Work with Hydra config system
4. **Zero Compromise on Quality** - Production-grade code only

**Inherited Assets:**
- ✅ OnDiskInductiveDataset (487 lines) - Sequential processing
- ✅ OnDiskTransductiveDataset (388 lines) - Large graph handling
- ✅ SQLiteIndexBackend (317 lines) - Structure indexing
- ✅ 65 passing tests
- ✅ Small/medium validation (PROTEINS, 5K nodes)

**The Gap:**
- ⚠️ No large-scale proof (30K-50K+ nodes)
- ⚠️ SQLite overhead (~500-1000 MB)
- ⚠️ No end-to-end training demonstration

---

## 🏗️ THREE-PHASE ARCHITECTURE

### Sacred Phase Ordering

```
Phase A → Phase B → Phase C
  ↓          ↓          ↓
PROOF    SURVIVAL   OPTIMIZATION
```

**Rationale:**
- **Phase A:** Easiest - proves disk-based iteration (multiple graphs)
- **Phase B:** Harder - proves OnDisk indexing at scale (single large graph)
- **Phase C:** Only after proof - optimize SQLite for speed/memory

**Anti-Patterns:**
- ❌ Optimizing before proving functionality
- ❌ Faking baseline failures
- ❌ Implementing without proper Hydra configs

---

## 🎯 PHASE A: INDUCTIVE GIANT

**Goal:** Prove OnDiskInductiveDataset enables training on 50K+ graphs where in-memory OOMs.

### A.1: Dataset Generation

**Create:** `scripts/generate_giant_inductive.py`

```python
"""Generate 50K small graphs for scale testing."""
def generate_giant_inductive(
    num_graphs=50000,      # Large enough to exceed RAM
    nodes_per_graph=100,   # Small enough for fast processing  
    avg_degree=5,
    num_classes=10,
    output_dir="./data/giant_inductive"
):
    # Expected: ~500 MB raw, 8-12 GB after lifting in-memory
    # OnDisk: <500 MB constant
    pass
```

### A.2: Hydra Configs

**`configs/experiment/prove_inductive.yaml`:**
```yaml
# @package _global_

defaults:
  - override /dataset: graph/manual_graph
  - override /model: simplicial/sccnn_custom
  - override /transforms: liftings/graph2simplicial/feature_simplex
  - override /trainer: default

tags: ["ondisk", "inductive", "giant", "proof"]

dataset:
  loader:
    _target_: topobench.data.loaders.ManualGraphDatasetLoader
    name: giant_inductive
    data_dir: ${paths.data_dir}/giant_inductive
  parameters:
    task_level: graph
    num_classes: 10
  split_params:
    split_type: random
    train_prop: 0.7
    val_prop: 0.15
  # Enable OnDisk
  use_ondisk: true
  ondisk_config:
    force_reload: false

transforms:
  lifting:
    complex_dim: 2

model:
  feature_encoder:
    out_channels: 32
  backbone:
    n_layers: 2

trainer:
  max_epochs: 5
  accelerator: cpu

# Expected: <500 MB memory, successful training
```

**`configs/experiment/baseline_inductive.yaml`:** (Same but `use_ondisk: false`)

### A.3: Validation Script

**Create:** `scripts/validate_phase_a.py`

```python
"""Validate Phase A: Inductive Giant."""
import tracemalloc
import subprocess

def run_experiment(config_name):
    tracemalloc.start()
    result = subprocess.run(
        ["python", "topobench/run.py", f"experiment={config_name}"],
        capture_output=True, timeout=3600
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"success": result.returncode == 0, "peak_gb": peak / 1024**3}

def main():
    print("🚀 PHASE A VALIDATION: INDUCTIVE GIANT\n")
    
    # Generate dataset
    # subprocess.run(["python", "scripts/generate_giant_inductive.py"])
    
    # Test baseline
    baseline = run_experiment("baseline_inductive")
    
    # Test OnDisk
    ondisk = run_experiment("prove_inductive")
    
    # Compare
    if not baseline["success"] and ondisk["success"]:
        print("✓ PHASE A SUCCESS: OnDisk enables impossible scale!")
    elif baseline["success"] and ondisk["success"]:
        ratio = ondisk["peak_gb"] / baseline["peak_gb"]
        print(f"✓ PHASE A SUCCESS: OnDisk uses {100*(1-ratio):.0f}% less memory!")
```

### A.4: Success Criteria

**Must Have:**
- ✅ Baseline fails (OOM) OR uses >8 GB
- ✅ OnDisk succeeds with <500 MB
- ✅ Training completes 1+ epochs

---

## 🎯 PHASE B: TRANSDUCTIVE TITAN

**Goal:** Prove OnDiskTransductiveDataset + SQLite enables training on 50K node graph.

### B.1: Dataset Generation

**Create:** `scripts/generate_titan_graph.py`

```python
"""Generate single 50K node graph with high clustering."""
def generate_titan_graph(
    num_nodes=50000,
    avg_degree=30,
    num_features=64,
    num_classes=10
):
    # Expected: ~750K edges, 2-5M triangles
    # In-memory: 10-20 GB
    # OnDisk: <2 GB
    pass
```

### B.2: Hydra Config

**`configs/experiment/prove_transductive.yaml`:**
```yaml
# @package _global_

defaults:
  - override /dataset: graph/manual_graph
  - override /model: simplicial/sccnn_custom
  - override /transforms: liftings/graph2simplicial/feature_simplex
  - override /trainer: default

tags: ["ondisk", "transductive", "titan", "indexed"]

dataset:
  loader:
    _target_: topobench.data.loaders.ManualGraphDatasetLoader
    name: titan_graph
    data_dir: ${paths.data_dir}/titan_graph
  parameters:
    task_level: node
    num_classes: 10
  split_params:
    split_type: node_split
    train_prop: 0.6
  use_ondisk: true
  ondisk_config:
    index_backend: sqlite
    batch_query_enabled: true

transforms:
  lifting:
    complex_dim: 2

model:
  feature_encoder:
    out_channels: 64
  backbone:
    n_layers: 3

trainer:
  max_epochs: 10

# Expected: <2 GB, <100ms query latency
```

### B.3: Validation Script

**Create:** `scripts/validate_phase_b.py`

```python
"""Validate Phase B: Transductive Titan."""
import time
from topobench.data.index import SQLiteIndexBackend

def test_query_performance(index_path, num_queries=1000):
    backend = SQLiteIndexBackend(data_dir=index_path)
    backend.open()
    
    query_times = []
    for _ in range(num_queries):
        batch_nodes = random.sample(range(50000), k=32)
        start = time.perf_counter()
        backend.query_by_nodes(batch_nodes)
        query_times.append(time.perf_counter() - start)
    
    avg_ms = sum(query_times) / len(query_times) * 1000
    return avg_ms < 100, avg_ms

def main():
    print("🚀 PHASE B VALIDATION: TRANSDUCTIVE TITAN\n")
    
    # Generate + test
    # Similar to Phase A but with query latency checks
```

### B.4: Success Criteria

**Must Have:**
- ✅ OnDisk indexing completes
- ✅ Query latency <100ms average
- ✅ Training completes 1+ epochs
- ✅ Memory <2 GB constant

---

## 🎯 PHASE C: OPTIMIZATION LAP

**Goal:** Optimize SQLite ONLY after Phases A & B prove functionality.

### C.1: SQLite Optimizations

**Modify:** `topobench/data/index/sqlite_backend.py` (line 97-100)

**Current:**
```python
self.conn = sqlite3.connect(str(self.db_path), **self.kwargs)
self.conn.execute("PRAGMA journal_mode=WAL")
self.conn.execute("PRAGMA synchronous=NORMAL")
```

**Optimized:**
```python
self.conn = sqlite3.connect(str(self.db_path), **self.kwargs)
# Aggressive read-heavy optimizations
self.conn.execute("PRAGMA journal_mode=OFF")
self.conn.execute("PRAGMA synchronous=OFF")
self.conn.execute("PRAGMA cache_size=-64000")  # 64 MB
self.conn.execute("PRAGMA temp_store=MEMORY")
self.conn.execute("PRAGMA mmap_size=268435456")  # 256 MB mmap
```

**Why Safe:**
- Building static index (no rollback needed)
- Single-threaded access during training
- Can rebuild if corruption (cached data)

### C.2: Batch Commits

Add method to `SQLiteIndexBackend`:

```python
def insert_batch_optimized(self, structures, commit_interval=10000):
    """Insert with periodic commits to reduce memory."""
    batch = []
    count = 0
    
    self.conn.execute("BEGIN TRANSACTION")
    for struct_id, nodes in structures:
        batch.append((struct_id, nodes))
        count += 1
        
        if count % commit_interval == 0:
            self._flush_batch(batch)
            batch = []
            self.conn.commit()
            self.conn.execute("BEGIN TRANSACTION")
    
    if batch:
        self._flush_batch(batch)
    self.conn.commit()
```

### C.3: Before/After Validation

**Create:** `scripts/validate_phase_c.py`

```python
"""Compare SQLite before/after optimization."""
def compare_versions():
    # Test on 10K node graph
    mem_before, time_before = benchmark(optimized=False)
    mem_after, time_after = benchmark(optimized=True)
    
    print(f"Memory: {mem_before:.0f} MB → {mem_after:.0f} MB ({100*(1-mem_after/mem_before):.0f}% reduction)")
    print(f"Time: {time_before:.1f}s → {time_after:.1f}s ({time_before/time_after:.1f}x speedup)")
```

### C.4: Success Criteria

**Must Have:**
- ✅ Memory reduction >20%
- ✅ All tests still pass
- ✅ Query latency maintained

---

## 📊 INTEGRATION WITH TOPOBENCH

### Preprocessor Integration

**Modify:** `topobench/data/preprocessor/preprocessor.py`

```python
def load_dataset_splits(self, split_params):
    """Load splits with optional OnDisk processing."""
    
    if self.cfg.get("use_ondisk", False):
        ondisk_cfg = self.cfg.get("ondisk_config", {})
        
        if self.dataset.parameters.task_level == "graph":
            from topobench.data.preprocessor import OnDiskInductiveDataset
            dataset = OnDiskInductiveDataset(
                dataset=self.dataset,
                data_dir=self.dataset_dir,
                transforms_config=self.transforms_config,
                **ondisk_cfg
            )
        elif self.dataset.parameters.task_level == "node":
            from topobench.data.preprocessor import OnDiskTransductiveDataset
            dataset = OnDiskTransductiveDataset(
                dataset=self.dataset,
                data_dir=self.dataset_dir,
                transforms_config=self.transforms_config,
                **ondisk_cfg
            )
        
        return self._split_ondisk_dataset(dataset, split_params)
    else:
        return self._load_standard_splits(split_params)
```

### Config Schema

All experiments use this pattern:

```yaml
dataset:
  # ... standard config ...
  use_ondisk: false  # Toggle
  ondisk_config:
    force_reload: false
    index_backend: sqlite  # For transductive
```

---

## 🧪 TESTING STRATEGY

### Test Hierarchy

```
Unit Tests (existing ✓)
    ↓
Integration Tests (create)
    ↓
Phase A Validation
    ↓
Phase B Validation
    ↓
Phase C Optimization
    ↓
End-to-End Training
```

### New Integration Tests

**Create:** `test/integration/test_ondisk_training.py`

```python
def test_ondisk_inductive_small():
    """Test OnDisk with PROTEINS."""
    
def test_ondisk_transductive_small():
    """Test OnDisk with 1K node graph."""
    
def test_ondisk_config_integration():
    """Test Hydra config loading."""
    
def test_ondisk_dataloader():
    """Test batching/shuffling."""
```

---

## 📈 SUCCESS METRICS

### Phase A (Inductive)

| Metric | Baseline | OnDisk | Target |
|--------|----------|--------|--------|
| Peak Memory | >8 GB or OOM | <500 MB | <1 GB |
| Training | No | Yes | Yes |

### Phase B (Transductive)

| Metric | Baseline | OnDisk | Target |
|--------|----------|--------|--------|
| Peak Memory | >15 GB or OOM | <2 GB | <3 GB |
| Query Latency | N/A | <100 ms | <100 ms |
| Training | No | Yes | Yes |

### Phase C (Optimization)

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Index Memory | ~800 MB | ? | <500 MB |
| Query Time | ~80 ms | ? | <50 ms |

---

## 🚀 EXECUTION CHECKLIST

### Phase A (Week 1)
- [ ] Create `scripts/generate_giant_inductive.py`
- [ ] Create `configs/experiment/prove_inductive.yaml`
- [ ] Create `configs/experiment/baseline_inductive.yaml`
- [ ] Create `scripts/validate_phase_a.py`
- [ ] Run validation, document results
- [ ] Generate `PHASE_A_REPORT.md`

### Phase B (Week 1-2)
- [ ] Create `scripts/generate_titan_graph.py`
- [ ] Create `configs/experiment/prove_transductive.yaml`
- [ ] Create `scripts/validate_phase_b.py`
- [ ] Run validation, document results
- [ ] Generate `PHASE_B_REPORT.md`

### Phase C (Week 2)
- [ ] Modify `sqlite_backend.py` with PRAGMAs
- [ ] Implement `insert_batch_optimized()`
- [ ] Create `scripts/validate_phase_c.py`
- [ ] Run before/after comparison
- [ ] Generate `PHASE_C_REPORT.md`

### Integration
- [ ] Modify `preprocessor.py` for OnDisk loading
- [ ] Create `test/integration/test_ondisk_training.py`
- [ ] Run full test suite
- [ ] Update documentation

### Final Deliverables
- [ ] `FINAL_VALIDATION_REPORT.md`
- [ ] Clean up temporary files
- [ ] Update README with OnDisk usage
- [ ] Submit PR

---

## 💡 DECISION FRAMEWORK

**When to move to next phase:**
- ✅ All success criteria met
- ✅ Tests passing
- ✅ Documentation complete

**When to pivot:**
- ⚠️ If Phase A/B fail after 2 attempts → Document existing evidence
- ⚠️ If SQLite can't be optimized → Keep working version, document limitations

**When to ask for help:**
- ❓ Tests fail unexpectedly
- ❓ Performance worse than expected
- ❓ Config integration unclear

---

## 📞 KEY FILES REFERENCE

**Core Implementation:**
- `topobench/data/preprocessor/ondisk_inductive.py`
- `topobench/data/preprocessor/ondisk_transductive.py`
- `topobench/data/index/sqlite_backend.py`
- `topobench/data/structure_detection.py`

**Config System:**
- `topobench/run.py` (entry point)
- `configs/run.yaml` (default config)
- `configs/experiment/` (experiment configs)

**Test Suite:**
- `test/data/preprocessor/test_ondisk_inductive.py`
- `test/data/preprocessor/test_ondisk_transductive.py`

---

**Status:** Ready for execution  
**Estimated Time:** 2-3 days for Phases A+B, 0.5 days for Phase C  
**Confidence:** 95% - Strong foundation, clear path

🚀 **Let's make the impossible possible!**
