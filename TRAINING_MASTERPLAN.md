# 🎯 TRAINING MASTERPLAN: Large-Scale Topological Deep Learning

**Date:** November 20, 2025, 22:30 UTC+01:00  
**Objective:** Demonstrate OnDisk infrastructure on datasets that REQUIRE it (OOM without it)  
**Status:** 🎯 **PLANNING PHASE**

---

## 🚀 MISSION STATEMENT

**Prove that OnDiskTransductiveDataset enables topological deep learning on graphs that are impossible to process in-memory.**

**Target Dataset:** `ogbn-products` (2.4M nodes, 61M edges)  
**Challenge:** Finding all triangles in-memory → **MEMORY EXPLOSION** 💥  
**Solution:** OnDiskTransductiveDataset with offline indexing → **Constant memory** ✅

---

## 📊 THE PROBLEM (Why This Matters)

### In-Memory Approach (Traditional)

```python
# Load graph
graph = load_ogbn_products()  # 2.4M nodes, 61M edges → ~500 MB

# Find ALL triangles
triangles = enumerate_all_triangles(graph)  # PROBLEM!
# Estimated: 100M - 1B+ triangles
# Memory: 10-50 GB+ → OOM CRASH 💥
```

**Result:** Researchers cannot use topological features on production-scale graphs.

### OnDisk Approach (Ours)

```python
# Build offline index (one-time, constant memory)
dataset = OnDiskTransductiveDataset(
    graph_data=graph,
    data_dir="./index",
    max_structure_size=3,  # triangles
)
dataset.build_index()  # Memory: ~2 GB constant ✅

# Query batches during training (fast)
batch = [node_0, node_1, ..., node_1000]
triangles = dataset.query_batch(batch)  # <100ms ✅
```

**Result:** Researchers CAN use topological features on any graph!

---

## 🎯 MISSION OBJECTIVES

### Phase 1: Infrastructure Validation (Priority: CRITICAL)
**Goal:** Prove OnDiskTransductiveDataset works on ogbn-products

**Tasks:**
1. ✅ Load ogbn-products dataset
2. ✅ Build offline triangle index with constant memory
3. ✅ Verify correctness (sample triangles, compare to baseline)
4. ✅ Measure performance (indexing time, query time, memory)

**Success Criteria:**
- [ ] Index builds successfully (< 2 GB RAM during indexing)
- [ ] Query performance: <100ms for 1K-node batches
- [ ] Correctness: 100% match with baseline (on sample)

**Estimated Time:** 30-60 minutes
**Risk:** Low (already validated on 5K-node synthetic graph)

---

### Phase 2: Training Pipeline (Priority: HIGH)
**Goal:** Train a topological model on ogbn-products using OnDisk

**Tasks:**
1. ✅ Create training script following TopoBench standards
2. ✅ Integrate OnDiskTransductiveDataset with TBDataloader
3. ✅ Implement baseline (GCN/SAGE without topological features)
4. ✅ Implement topological model (SCNN/SCCNN with triangle features)
5. ✅ Train both models (10-20 epochs)
6. ✅ Compare performance (accuracy, memory, time)

**Success Criteria:**
- [ ] Training runs without OOM
- [ ] Memory stays < 4 GB throughout training
- [ ] Model converges (loss decreases)
- [ ] Results documented

**Estimated Time:** 2-4 hours (depending on training time)
**Risk:** Medium (complex integration, long training)

---

### Phase 3: Demonstration & Documentation (Priority: MEDIUM)
**Goal:** Document and present results for submission

**Tasks:**
1. ✅ Create validation report
2. ✅ Update MASTER_PLAN.md
3. ✅ Write submission narrative
4. ✅ Create visualizations (memory usage, performance comparison)

**Success Criteria:**
- [ ] Professional documentation
- [ ] Clear value proposition
- [ ] Submission-ready materials

**Estimated Time:** 30-60 minutes
**Risk:** Low (documentation task)

---

## 📊 DATASET ANALYSIS: ogbn-products

### Specifications
- **Name:** ogbn-products
- **Domain:** Product co-purchasing network (Amazon)
- **Task:** Node classification (47 categories)
- **Nodes:** 2,449,029
- **Edges:** 61,859,140 (undirected)
- **Features:** 100 per node
- **Split:** 196,615 train / 39,323 val / 2,213,091 test

### Topological Characteristics
- **Average Degree:** ~50
- **Estimated Triangles:** 100M - 1B+ (highly variable)
- **Clustering Coefficient:** Unknown (large)

### Memory Requirements

**In-Memory Baseline:**
- Graph loading: ~500 MB
- Node features: ~1 GB (2.4M × 100 × 4 bytes)
- Triangle enumeration: **10-50 GB+** (estimated)
- **TOTAL: 11-51+ GB → OOM on most machines** ❌

**OnDisk Approach:**
- Graph loading: ~500 MB
- Indexing: ~2 GB constant (streaming enumeration)
- Disk storage: ~5-20 GB (compressed bitmap index)
- **TOTAL: < 3 GB RAM** ✅

---

## 🛠 TECHNICAL APPROACH

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Offline Indexing (One-Time)                       │
│  ┌────────────┐      ┌──────────────────┐     ┌─────────┐ │
│  │ ogbn-      │  →   │ OnDisk           │  →  │ SQLite  │ │
│  │ products   │      │ Transductive     │     │ Index   │ │
│  │ (2.4M)     │      │ Dataset          │     │ (disk)  │ │
│  └────────────┘      └──────────────────┘     └─────────┘ │
│      500 MB              ~2 GB RAM              5-20 GB    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Training Loop (Fast Queries)                      │
│  ┌────────────┐      ┌──────────────────┐     ┌─────────┐ │
│  │ Mini-Batch │  →   │ Query Triangles  │  →  │ SCNN    │ │
│  │ (1K nodes) │      │ from Index       │     │ Model   │ │
│  │            │      │ (<100ms)         │     │         │ │
│  └────────────┘      └──────────────────┘     └─────────┘ │
│      <100 MB             <100ms                 ~1 GB      │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **OnDiskTransductiveDataset**
   - Offline triangle indexing with StreamingCliqueEnumerator
   - SQLite + PyRoaring storage
   - Fast batch queries

2. **Training Script** (TopoBench-style)
   - Baseline: GCN/GraphSAGE (graph-only)
   - Topological: SCNN with triangle features
   - Memory monitoring throughout
   - Standard metrics (accuracy, loss, time)

3. **Integration Points**
   - Use existing TopoBench loaders for ogbn-products
   - Integrate OnDiskTransductiveDataset as data source
   - Follow TopoBench training conventions
   - Use standard configs/models when possible

---

## 📝 IMPLEMENTATION PLAN

### Step 1: Validate Infrastructure (30-60 min)

**Script:** `validate_ogbn_ondisk.py`

```python
# Load ogbn-products
from ogb.nodeproppred import NodePropPredDataset
dataset = NodePropPredDataset('ogbn-products')

# Build OnDisk index
ondisk = OnDiskTransductiveDataset(
    graph_data=convert_to_pyg(dataset),
    data_dir="./data/ogbn_products_index",
    max_structure_size=3,  # triangles
)

# Monitor memory during indexing
monitor_memory(ondisk.build_index)

# Test queries
batch = sample_nodes(1000)
triangles = ondisk.query_batch(batch)

# Verify correctness
baseline_triangles = networkx_triangles(graph, batch)
assert triangles == baseline_triangles
```

**Output:**
- Memory profile during indexing
- Indexing time
- Query performance
- Correctness validation

---

### Step 2: Training Pipeline (2-4 hours)

**Script:** `train_ogbn_topological.py` (TopoBench-style)

```python
# Standard TopoBench imports
from topobench.data.loaders import OGBNodeLoader
from topobench.data.preprocessor import OnDiskTransductiveDataset
from topobench.nn.backbones.graph import GCN
from topobench.nn.backbones.simplicial import SCNN
from topobench.trainer import TBTrainer

# Load dataset
loader = OGBNodeLoader(name='ogbn-products')
graph = loader.load()

# Build OnDisk index
ondisk = OnDiskTransductiveDataset(
    graph_data=graph,
    data_dir="./data/ogbn_products_index",
    max_structure_size=3,
)

# Baseline: GCN (graph-only)
baseline_model = GCN(
    in_channels=100,
    hidden_channels=256,
    out_channels=47,
    num_layers=3,
)

# Train baseline
trainer = TBTrainer(model=baseline_model, ...)
baseline_results = trainer.train()

# Topological: SCNN (with triangles)
topo_model = SCNN(
    in_channels=100,
    hidden_channels=256,
    out_channels=47,
    use_triangles=True,
    ondisk_dataset=ondisk,  # Query triangles on-the-fly
)

# Train topological
trainer = TBTrainer(model=topo_model, ...)
topo_results = trainer.train()

# Compare
print(f"Baseline: {baseline_results['test_acc']:.4f}")
print(f"Topological: {topo_results['test_acc']:.4f}")
```

**Output:**
- Training logs
- Accuracy comparison
- Memory profiles
- Time measurements

---

### Step 3: Documentation (30-60 min)

**Documents to Create:**
1. `OGBN_PRODUCTS_VALIDATION.md` - Technical validation report
2. `TRAINING_RESULTS.md` - Training results and analysis
3. `SUBMISSION_NARRATIVE.md` - Final submission story

**Update Existing:**
1. `MASTER_PLAN.md` - Add ogbn-products validation
2. `MISSIONS_COMPLETE.md` - Update with new results

---

## ⚠️ RISK ASSESSMENT

### Critical Risks

1. **Indexing Time Too Long**
   - **Risk:** Building triangle index takes hours
   - **Mitigation:** Use smaller subgraph first, then scale
   - **Fallback:** Document indexing process, show it's one-time cost

2. **Training Integration Complex**
   - **Risk:** Integrating OnDisk with TopoBench training is non-trivial
   - **Mitigation:** Start with simple baseline, iterate
   - **Fallback:** Demonstrate infrastructure, show training is future work

3. **Memory Still Too High**
   - **Risk:** Even OnDisk exceeds available RAM
   - **Mitigation:** Use smaller batches, optimize queries
   - **Fallback:** Document what's needed, show it's better than in-memory

### Medium Risks

1. **Query Performance Slow**
   - **Risk:** <100ms target not met
   - **Mitigation:** Optimize SQLite queries, use smaller batches
   - **Fallback:** Document performance, show it's still usable

2. **Accuracy Lower Than Baseline**
   - **Risk:** Topological features don't help
   - **Mitigation:** Hyperparameter tuning
   - **Fallback:** Focus on infrastructure value, not accuracy

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable Success (Must Have)
- [ ] OnDisk index builds on ogbn-products
- [ ] Memory stays < 4 GB during indexing
- [ ] Query performance documented
- [ ] Correctness validated (sample)

**Achievement:** Infrastructure validated on production-scale graph ✅

### Target Success (Should Have)
- [ ] Training pipeline works
- [ ] Baseline model trains successfully
- [ ] Memory profile documented throughout training
- [ ] Results compared (baseline vs topological)

**Achievement:** End-to-end demonstration ✅

### Stretch Success (Nice to Have)
- [ ] Topological model beats baseline
- [ ] Full training to convergence
- [ ] Comprehensive benchmarks
- [ ] Publication-quality results

**Achievement:** SOTA performance + infrastructure ✅

---

## 📊 TIMELINE

### Immediate (Tonight, 2-4 hours)
1. **Phase 1:** Validate infrastructure (1 hour)
   - Load ogbn-products
   - Build OnDisk index
   - Test queries
   - Verify correctness

2. **Decision Point:** Continue to training or document validation?

### Short-term (If continuing)
3. **Phase 2:** Training pipeline (2-3 hours)
   - Create training script
   - Train baseline
   - Train topological model
   - Compare results

4. **Phase 3:** Documentation (1 hour)
   - Write reports
   - Update MASTER_PLAN
   - Prepare submission

### Total Estimated Time: 4-7 hours

---

## 💡 STRATEGIC DECISION

### Option A: Full Training (Ambitious)
**Pros:**
- Complete demonstration
- End-to-end validation
- Strongest submission

**Cons:**
- Time-intensive (4-7 hours)
- Risk of incomplete results
- Training may not converge quickly

**Recommendation:** Only if we have time and high confidence

---

### Option B: Infrastructure Validation Only (Pragmatic) ✅ RECOMMENDED
**Pros:**
- Achievable in 1-2 hours
- Low risk
- Proves core value proposition
- Can add training later if needed

**Cons:**
- Doesn't show full pipeline
- No accuracy comparison

**Recommendation:** **START HERE** - validate infrastructure first, then decide

---

## 🚀 EXECUTION PLAN (RECOMMENDED)

### Phase 1: Infrastructure Validation (START HERE)
**Time:** 1-2 hours  
**Risk:** Low  
**Value:** High

**Steps:**
1. Create `validate_ogbn_ondisk.py`
2. Load ogbn-products and convert to PyG
3. Build OnDiskTransductiveDataset index
4. Monitor memory (prove constant)
5. Test batch queries (prove fast)
6. Sample-verify correctness
7. Document results

**Deliverable:** `OGBN_PRODUCTS_INFRASTRUCTURE_VALIDATION.md`

### Decision Point: Continue?
**If validation succeeds AND time permits:**
- Proceed to Phase 2 (Training)

**If validation succeeds BUT time limited:**
- Document validation
- Mark as "infrastructure proven, training future work"
- **Still a HUGE WIN for submission** ✅

### Phase 2: Training (OPTIONAL)
**Time:** 2-4 hours  
**Risk:** Medium  
**Value:** Very High

Only proceed if:
- [ ] Phase 1 completed successfully
- [ ] We have 3+ hours available
- [ ] High confidence in training setup

---

## 📈 VALUE PROPOSITION

### What This Proves (Infrastructure Only)
✅ OnDisk works on 2.4M-node graphs  
✅ Constant memory (vs OOM for in-memory)  
✅ Fast queries (<100ms)  
✅ Production-scale validation  
✅ **Enables research that wasn't possible**

**Submission Strength:** Very Strong (proves necessity)

### What This Proves (Infrastructure + Training)
✅ All of the above, PLUS:  
✅ End-to-end pipeline works  
✅ Topological models trainable at scale  
✅ Real performance comparison  
✅ **Complete demonstration**

**Submission Strength:** Exceptional (proves value + necessity)

---

## 🎯 RECOMMENDATION

**Start with Phase 1 (Infrastructure Validation):**
1. Achievable in 1-2 hours
2. Low risk
3. High value
4. Proves core claim

**Decision after Phase 1:**
- Success + Time → Proceed to training
- Success + Limited time → Document and submit
- Issues → Debug and document learnings

**Either way, we WIN:**
- Infrastructure validated on production-scale
- Proof that OnDisk is **necessary** not just nice
- Strongest possible submission position

---

## 🚀 NEXT STEP

**IMMEDIATE ACTION:** Execute Phase 1 (Infrastructure Validation)

**Command to start:**
```bash
# Create validation script
python validate_ogbn_ondisk.py
```

**Expected outcome:**
- Memory profile showing constant usage
- Query performance metrics
- Correctness validation
- **Proof that OnDisk enables production-scale topological DL**

---

**Status:** 🎯 **READY TO EXECUTE**  
**Phase:** Phase 1 (Infrastructure Validation)  
**Estimated Time:** 1-2 hours  
**Confidence:** High ✅

**Let's prove OnDisk is NECESSARY, not just nice! 🚀**
