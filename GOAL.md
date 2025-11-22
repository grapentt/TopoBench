# GOAL: TopoBench Challenge B1 & B1 Bonus - Comprehensive Analysis

**Last Updated**: 2024-11-21

---

## Executive Summary

Core on-disk functionality implemented but integration incomplete. Main gaps: seamless PreProcessor integration, transductive training integration, production validation scripts, and real-world dataset implementations.

**Status**: ~60% complete. Critical path: 16-22 hours remaining.

---

## 1. What Exists

### 1.1 Core Infrastructure ✓

**OnDiskInductiveDataset** (`topobench/data/preprocessor/ondisk_inductive.py`, 472 lines)
- Sequential disk-backed processing, O(1) memory per sample
- Transform caching via parameter hashing
- `load_dataset_splits` method for train/val/test
- **Test Coverage**: 29 unit tests (512 lines) - comprehensive ✓

**OnDiskTransductiveDataset** (`topobench/data/preprocessor/ondisk_transductive.py`, 357 lines)
- Indexes topological structures offline  
- On-demand structure querying via `StructureQueryEngine`
- O(1) memory regardless of graph size
- **Test Coverage**: None - needs implementation ❌

**Structure Detection** (`topobench/data/structure_detection.py`, 280 lines)
- Streaming clique/triangle enumeration, O(k) memory
- Optimized for triangles: O(n*d²) complexity
- `build_clique_index` for persistent indexing

**Structure Query Engine** (`topobench/data/structure_query.py`, 321 lines)
- High-level query interface with SQLite backend
- `query_batch` for batch queries
- `verify_query_correctness` for validation

### 1.2 Synthetic Datasets & Loaders ✓

**Datasets**:
- `SyntheticLargeInductiveDataset` - many graphs, configurable size
- `SyntheticLargeTransductiveDataset` - single large graph
- Both inherit from `InMemoryDataset` (correct - generate raw data)

**Loaders**:
- `SyntheticLargeInductiveLoader` - inherits `AbstractLoader`
- `SyntheticLargeTransductiveLoader` - inherits `AbstractLoader`

### 1.3 Validation Scripts (Root Dir) ⚠️

Four scripts exist but need major refinement:
1. `test_1_1_inmemory_inductive_FAILS_PROPER.py` - shows in-memory OOM
2. `test_1_2_ondisk_inductive_WORKS_PROPER.py` - shows on-disk success
3. `test_2_1_transductive_FAILS_PROPER_V2.py` - shows in-memory OOM
4. `test_2_2_transductive_WORKS_PROPER_V2.py` - shows on-disk success

**Issues**:
- Not TopoBench-style workflow (direct API usage)
- No guaranteed OOM (tunable params but no calculation)
- Transductive training integration incomplete

### 1.4 Tutorial Notebooks ⚠️

- `tutorial_ondisk_inductive.ipynb` - good foundation, needs polish
- `tutorial_ondisk_transductive.ipynb` - needs checking

---

## 2. What Works

✅ **OnDiskInductiveDataset**: Core functionality validated with 29 tests
✅ **Structure detection**: Streaming algorithms work correctly  
✅ **Structure querying**: Index building and queries validated
✅ **Synthetic data generation**: Both inductive/transductive work

---

## 3. What's Missing (Critical Issues)

### 3.1 Architecture Gaps

❌ **Seamless PreProcessor Integration**

Current (confusing):
```python
# In-memory
preprocessor = PreProcessor(dataset, dir, transforms)

# On-disk (different API!)
ondisk = OnDiskInductiveDataset(dataset, dir, transforms)
```

Desired (seamless):
```python
# Unified interface
preprocessor = create_preprocessor(dataset, dir, transforms, mode="ondisk")
```

**Solution**: Factory pattern in `topobench/data/preprocessor/__init__.py`

❌ **Transform System Validation**
- `OnDiskInductiveDataset` instantiates transforms but needs validation
- Must test with all TopoBench liftings (SimplicialCliqueLifting, etc.)
- Verify output matches `PreProcessor`

❌ **Transductive Training Integration**
- Query system works but training integration incomplete
- Need custom dataloader/collate for on-demand structure querying
- Current validation script uses simplified demo (line 240-252 in test_2_2)

### 3.2 Missing Components

❌ **OGBN-products dataset** (B1 Bonus requirement)
- Loader not implemented
- Integration with `OnDiskTransductiveDataset` needed
- Training script needed

❌ **Real-world inductive dataset** (B1 requirement)
- Dataset not selected (candidates: OGBG-molhiv, ZINC, large TUDataset)
- Loader not implemented
- Training script needed

❌ **OnDiskTransductiveDataset unit tests**
- Zero tests currently
- Need 20-25 tests mirroring `test_ondisk_inductive.py`

❌ **Production validation scripts**
- Current scripts not rigorous enough
- Need automatic dataset size calculation: `params = calculate_size(MAX_RAM_GB)`
- Must guarantee OOM vs success with **same dataset**
- Must follow TopoBench workflow exactly (like `tutorial_model.ipynb`)

❌ **Integration tests**
- No end-to-end tests for full pipeline
- Need: Loader → PreProcessor → Splits → DataLoader → Training

❌ **Performance benchmarks**
- No memory tracking (need to prove O(1) memory)
- No disk usage tracking
- No preprocessing/training time comparison

---

## 4. Architecture Recommendations

### 4.1 Unified Interface (HIGH PRIORITY)

**Recommendation**: Factory pattern

```python
# topobench/data/preprocessor/__init__.py
def create_preprocessor(dataset, data_dir, transforms_config=None, 
                       mode="auto", **kwargs):
    """Factory for creating appropriate preprocessor.
    
    Args:
        mode: "inmemory" | "ondisk" | "auto"
            - "auto": detect based on dataset size
            - "inmemory": standard PreProcessor
            - "ondisk": OnDiskInductiveDataset or OnDiskTransductiveDataset
    """
    if mode == "ondisk" or _should_use_ondisk(dataset, mode):
        if _is_inductive(dataset):
            return OnDiskInductiveDataset(dataset, data_dir, transforms_config, **kwargs)
        else:
            return OnDiskTransductiveDataset(dataset, data_dir, transforms_config, **kwargs)
    else:
        return PreProcessor(dataset, data_dir, transforms_config, **kwargs)
```

**Benefits**: Backward compatible, clear separation, easy to document

### 4.2 Dataset Inheritance (RESOLVED ✓)

**Current** (correct):
- Synthetic datasets inherit `InMemoryDataset` (generate raw data)
- On-disk processors inherit `Dataset` (never load all into memory)

**Decision**: Keep current architecture

### 4.3 Transductive Training (HIGH PRIORITY)

**Solution**: Custom collate function for on-demand querying

```python
class OnDiskTransductiveCollate:
    def __init__(self, ondisk_dataset):
        self.ondisk_dataset = ondisk_dataset
    
    def __call__(self, node_ids_batch):
        # Query structures for this batch
        structures = self.ondisk_dataset.query_batch(node_ids_batch, fully_contained=True)
        # Build batch with only relevant structures
        return self._build_batch(node_ids_batch, structures)
```

Integrate with `TBDataloader` or create `OnDiskTransductiveDataloader` subclass

---

## 5. What Needs to Be Done

### Priority 1: Critical Path (Must Complete)

**Task 1.1: Unified Preprocessor Interface** ✅ COMPLETE
- Effort: 2.5 hours (completed 2024-11-21)
- ✅ Implemented factory pattern in `topobench/data/preprocessor/factory.py`
- ✅ Added mode detection logic (`_should_use_ondisk`, `_estimate_memory_requirement`)
- ✅ Wrote 22 tests (9 passing, 13 skipped with clear reasons)
- ✅ Updated exports in `__init__.py`
- ✅ MUTAG integration test validates real-world usage

**Task 1.2: Transform Validation** ✅ COMPLETE
- Effort: 2 hours (completed 2024-11-21)
- ✅ Tested SimplicialCliqueLifting (complex_dim 1, 2, 3)
- ✅ Tested HypergraphKHopLifting  
- ✅ Verified output matches `PreProcessor` on MUTAG/ENZYMES
- ✅ Added 10 integration tests (all passing)
- ✅ Created TRANSFORM_SUPPORT.md documentation

**Task 1.3: Transductive Training Integration** ✅ COMPLETE
- Effort: 3 hours (completed 2024-11-21)
- ✅ Created custom collate function (`OnDiskTransductiveCollate`)
- ✅ Created `NodeBatchSampler` for mini-batch sampling
- ✅ Tested mini-batch training workflow (17 tests passing)
- ✅ Verified memory stays constant (no accumulation)

**Task 1.4: Production Validation Scripts** ✅ COMPLETE
- Effort: 2.5 hours (completed 2024-11-21)
- ✅ Created reusable validation framework (`validation_utils.py`)
- ✅ Automated inductive validation script
- ✅ Automated transductive validation script
- ✅ Automatic dataset size calculation with `calculate_oom_params()`
- ✅ Memory tracking with `MemoryTracker`
- ✅ 10 utility tests (all passing)

**Task 1.5: OGBN-products Integration** ✅ COMPLETE
- Effort: 2 hours (completed 2024-11-21)
- ✅ Created `OGBNProductsLoader` (inherits `AbstractLoader`)
- ✅ Tested with `OnDiskTransductiveDataset` (3 tests passing)
- ✅ Created training script (`train_ogbn_products_ondisk.py`)
- ✅ Created comprehensive guide (`OGBN_PRODUCTS_GUIDE.md`)

**CRITICAL PATH TOTAL: 16-22 hours**

### Priority 2: High Priority (Should Complete)

**Task 2.1: OnDiskTransductiveDataset Unit Tests** ⭐⭐
- Effort: 2-3 hours
- Create `test/data/preprocessor/test_ondisk_transductive.py`
- 20-25 tests mirroring `test_ondisk_inductive.py`

**Task 2.2: Real Inductive Dataset** ⭐⭐
- Effort: 3-4 hours
- Research and select dataset (OGBG-molhiv, ZINC, or large TU)
- Create loader if needed
- Create training script

**Task 2.3: Refine Tutorial Notebooks** ⭐⭐
- Effort: 2-3 hours
- Update to reflect final architecture
- Add performance comparisons
- Professional formatting

**Task 2.4: Create GUIDE.md** ⭐⭐
- Effort: 2-3 hours
- User-facing documentation
- Quick starts, API reference, troubleshooting

**Task 2.5: Integration Tests** ⭐⭐
- Effort: 2-3 hours
- End-to-end pipeline tests
- Multiple models and liftings

**HIGH PRIORITY TOTAL: 11-16 hours**

### Priority 3: Medium Priority (Nice to Have)

**Task 3.1: Memory Tracking Utilities** ⭐
- Effort: 1-2 hours
- Utilities for profiling memory/disk usage

**Task 3.2: Performance Benchmarks** ⭐
- Effort: 2-3 hours
- Memory, preprocessing time, training speed benchmarks

**Task 3.3: Architecture Documentation** ⭐
- Effort: 1-2 hours
- Document design decisions and tradeoffs

**MEDIUM PRIORITY TOTAL: 4-7 hours**

---

## 6. Critical Issues Summary

### Blocking Issues (Must Fix)

1. **No seamless PreProcessor integration** - users face two different APIs
2. **Transductive training incomplete** - query system works but training integration missing
3. **Validation scripts not rigorous** - don't guarantee OOM, don't follow TopoBench style
4. **Transform validation missing** - need to verify correctness with all liftings
5. **No OnDiskTransductiveDataset tests** - zero test coverage

### Missing Deliverables

1. **OGBN-products integration** (B1 Bonus requirement)
2. **Real-world inductive dataset** (B1 requirement)
3. **Production validation scripts** (4 scripts, TopoBench-style)
4. **Integration tests** (end-to-end pipeline)
5. **GUIDE.md** (user documentation)

---

## 7. Estimated Timeline

**Minimum for Submission** (Critical Path Only): 16-22 hours
**Recommended for Quality** (Critical + High Priority): 27-38 hours  
**Complete Implementation** (All Priorities): 31-45 hours

---

## 8. Next Steps

1. **Read this GOAL.md thoroughly** ✓
2. **Create LONGTERM.md** - vision, architecture, design principles
3. **Create SHORTTERM.md** - track last 2 done, next 1 to do
4. **Create GUIDE.md** - user-facing documentation
5. **Create PR_COMMIT.md** - track files for B1 and B1 Bonus PRs
6. **Wait for next prompt** before implementing

---

## 9. Key Insights

### What's Good ✓
- Core on-disk functionality is solid and well-tested (inductive)
- Structure detection algorithms are mathematically sound
- Synthetic datasets work well for validation
- Code quality is high (typed, documented, PEP8)

### What Needs Work ❌
- Integration with existing TopoBench workflow is incomplete
- Transductive on-disk training needs finishing
- Validation scripts need to be rigorous and TopoBench-style
- Real-world datasets not implemented yet
- No tests for transductive on-disk

### The Big Picture
We have strong **building blocks** but incomplete **integration**. The on-disk approach **works** but doesn't **feel** like TopoBench yet. Users should use it **exactly like** the in-memory approach (just one parameter different), and validation should **rigorously prove** OOM prevention.

**Target**: Production-grade extension that seamlessly extends TopoBench, with undeniable proof of capability.
