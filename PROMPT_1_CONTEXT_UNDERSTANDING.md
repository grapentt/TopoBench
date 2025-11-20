# 🎯 PROMPT 1: Context Understanding & Analysis

**Role:** Senior Machine Learning Infrastructure Engineer  
**Task:** Understand and validate existing OnDisk infrastructure for TopoBench

---

## 📋 **YOUR FIRST TASK: READ AND UNDERSTAND**

You are taking over a project that has implemented novel memory-efficient data loaders for topological deep learning. Before making any changes, you must thoroughly understand what exists.

### **CRITICAL: Read These Files First (In Order)**

1. **`HANDOFF_TO_NEXT_AGENT.md`** ← START HERE
   - Complete mission context
   - What's been accomplished
   - What needs to be done
   - Technical challenges and solutions
   - ~30 minutes to read and understand

2. **`READY_FOR_NEXT_AGENT.md`**
   - Quick summary
   - Current status
   - Key takeaways

3. **`CLEANUP_SUMMARY.md`**
   - Workspace organization
   - What files are important

### **WHAT WAS BUILT**

The previous agent implemented two novel data loaders:

**1. OnDiskInductiveDataset** (Multiple Small Graphs)
- **File:** `topobench/data/preprocessor/ondisk_inductive.py` (487 lines)
- **Purpose:** Process many graphs sequentially without loading all into memory
- **Status:** ✅ Complete, 21 tests passing
- **Validation:** ✅ Proven on PROTEINS (1,113 graphs)
- **Memory:** Constant O(1) - proven with measurements

**2. OnDiskTransductiveDataset** (Single Large Graph)
- **File:** `topobench/data/preprocessor/ondisk_transductive.py` (388 lines)
- **Purpose:** Index topological structures (triangles) offline for large graphs
- **Backend:** SQLite (not RocksDB - this is important!)
- **Status:** ✅ Implemented and tested
- **Validation:** ✅ Proven on 5K-node, 10K-node graphs
- **Memory:** Constant during indexing, fast batch queries

### **WHAT'S BEEN VALIDATED**

✅ **Tier 1: PROTEINS (1,113 graphs)**
```
In-Memory: +230.6 MB → extrapolates to +115 GB @ 500K graphs
OnDisk: +0.0 MB (constant)
Status: PROVEN - OnDisk clearly superior
Report: M1.6_VALIDATION_REPORT.md
```

✅ **Tier 2: 5K-node synthetic graph**
```
Triangles: 78,205 indexed
Memory: +70.1 MB (constant)
Query time: 89.4ms (<100ms target)
Correctness: 100% validated against baseline
Status: PROVEN - Production-ready
Report: M2_LARGE_SCALE_VALIDATION_REPORT.md
```

✅ **Tier 3: 10K-node synthetic graph**
```
Triangles: 325,922 enumerated
Memory: +0.5 MB (constant, optimized enumeration)
Time: 1.0 second
Status: PROVEN - Optimization works
Script: prove_ondisk_works_OPTIMIZED.py
```

### **WHAT'S INCOMPLETE**

⚠️ **Large-Scale Comparative Validation (30K-50K nodes)**

**The Gap:** Need to prove OnDisk succeeds where in-memory fails

**Current Issue:** 
- SQLite overhead makes OnDisk use MORE memory than simple triangle counting
- Need either: (a) optimize SQLite, or (b) fair comparison where in-memory stores data

**Files to examine:**
- `topobench/data/index.py` ← SQLite backend (needs optimization)
- `FINAL_PROOF_ONDISK_VS_INMEMORY.py` ← Comparison script (shows the issue)

### **TECHNICAL CONTEXT**

**Key Algorithms:**
```python
# Triangle enumeration (optimized for k=3 only)
topobench/data/structure_detection.py:enumerate_k_cliques_streaming()

# SQLite index backend (needs memory optimization)
topobench/data/index.py:SQLiteIndexBackend

# Query engine
topobench/data/structure_query.py:StructureQueryEngine
```

**Known Issues:**
1. ⚠️ SQLite memory overhead (~500-1000 MB)
2. ⚠️ Optimization only for k=3 (triangles), k>3 still slow
3. ⚠️ Minor triangle count mismatches (~100-500 difference)

---

## 🎯 **YOUR TASK AFTER READING**

Once you understand the context, respond with:

1. **Summary of what exists:**
   - List the two loaders and their status
   - List the three validated scales
   - Identify the gap (large-scale proof)

2. **Technical understanding:**
   - Explain why SQLite uses more memory
   - Explain the difference between "counting" vs "storing" triangles
   - Identify the unfair comparison issue

3. **Proposed validation strategy:**
   - How to optimize SQLite (specific PRAGMA statements)
   - How to create fair comparison (in-memory must store data)
   - Which graph sizes to test (30K, 50K nodes)
   - Expected results (in-memory fails, OnDisk succeeds)

**DO NOT START CODING UNTIL:**
- You've read all three handoff files
- You understand the existing implementations
- You can explain the current issue
- You have a clear plan to fix it

---

## ⚠️ **IMPORTANT CONSTRAINTS**

**DO:**
- ✅ Read all handoff documentation thoroughly
- ✅ Examine existing code before modifying
- ✅ Run existing tests to verify they pass
- ✅ Understand the SQLite overhead issue
- ✅ Plan before executing

**DON'T:**
- ❌ Start coding immediately
- ❌ Break existing tests (65 tests must pass)
- ❌ Use RocksDB (we use SQLite)
- ❌ Ignore the handoff documentation
- ❌ Run git commands

---

## 📊 **SUCCESS CRITERIA FOR THIS PROMPT**

You successfully complete this task when you can:

1. ✅ Explain what OnDiskInductiveDataset does
2. ✅ Explain what OnDiskTransductiveDataset does
3. ✅ List the three validated scales
4. ✅ Explain the SQLite overhead issue
5. ✅ Propose specific optimizations (with PRAGMA statements)
6. ✅ Design a fair comparison experiment
7. ✅ Estimate time to complete the validation (2-3 hours)

**Once you demonstrate this understanding, you'll be ready for Prompt 2: Execution.**

---

**Start by reading `HANDOFF_TO_NEXT_AGENT.md` now.**
