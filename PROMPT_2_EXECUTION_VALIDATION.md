# 🚀 PROMPT 2: Execution - Prove OnDisk Works at Scale

**Role:** Lead Benchmarking Engineer and Validation Specialist  
**Prerequisite:** You must have completed Prompt 1 and demonstrated understanding

---

## 🎯 **YOUR MISSION**

**Objective:** Prove that OnDisk infrastructure enables topological deep learning on large graphs where traditional in-memory approaches fail or use excessive memory.

**Success Means:**
- In-memory approach: ❌ Fails (OOM) or uses excessive memory (>3GB)
- OnDisk approach: ✅ Succeeds with constant memory (<1.5GB)
- **"Death vs. Survival" demonstrated with measurements**

---

## 📋 **VALIDATION STRATEGY: Three Phases**

### **Phase 1: Fix SQLite Memory Overhead** (30 minutes)

**Problem:** SQLite currently uses 500-1000 MB overhead, making OnDisk appear worse than in-memory counting.

**Solution:** Optimize SQLite with PRAGMA statements

**File to modify:** `topobench/data/index.py`

**Changes to make:**
```python
def open(self) -> None:
    """Open database connection with optimizations."""
    self.conn = sqlite3.connect(self.db_path)
    
    # CRITICAL OPTIMIZATIONS - Add these:
    self.conn.execute("PRAGMA journal_mode=OFF")      # Disable WAL (Write-Ahead Logging)
    self.conn.execute("PRAGMA synchronous=OFF")        # Disable sync (faster writes)
    self.conn.execute("PRAGMA cache_size=-64000")      # 64MB cache (from -2000 default)
    self.conn.execute("PRAGMA temp_store=MEMORY")      # Use memory for temp tables
    self.conn.execute("PRAGMA mmap_size=268435456")    # 256MB memory-mapped I/O
    self.conn.execute("PRAGMA page_size=4096")         # Optimize page size
    
    self.cursor = self.conn.cursor()
    self._ensure_tables_exist()

def insert_batch(self, structures_iterator, batch_size: int = 10000) -> None:
    """Insert structures in batches with commits."""
    batch = []
    count = 0
    
    for struct_id, nodes in structures_iterator:
        batch.append((struct_id, len(nodes), json.dumps(nodes)))
        count += 1
        
        if len(batch) >= batch_size:
            self.cursor.executemany(
                "INSERT OR REPLACE INTO structures (structure_id, size, nodes) VALUES (?, ?, ?)",
                batch
            )
            self.conn.commit()  # Commit every batch
            batch = []
    
    # Insert remaining
    if batch:
        self.cursor.executemany(
            "INSERT OR REPLACE INTO structures (structure_id, size, nodes) VALUES (?, ?, ?)",
            batch
        )
        self.conn.commit()
```

**Test:** Run on 10K graph, verify memory drops from ~150MB to <100MB

---

### **Phase 2: Create Fair In-Memory Comparison** (30 minutes)

**Problem:** Current comparison counts triangles without storing them. OnDisk stores them for querying. Not a fair comparison!

**Solution:** Make in-memory actually store all triangles like OnDisk does.

**Create:** `prove_ondisk_fair_comparison.py`

**In-Memory Approach (Fair):**
```python
def inmemory_with_storage(graph):
    """In-memory approach that STORES all triangles (fair comparison)."""
    triangles_dict = {}  # Store like OnDisk does
    structure_id = 0
    
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if graph.has_edge(n1, n2):
                    triangle = tuple(sorted([node, n1, n2]))
                    if triangle[0] == node:
                        triangles_dict[structure_id] = {
                            'nodes': triangle,
                            'size': 3
                        }
                        structure_id += 1
    
    return triangles_dict  # This stays in memory!
```

**OnDisk Approach (Same as before):**
```python
def ondisk_with_storage(graph, index_dir):
    """OnDisk approach with SQLite storage."""
    data = from_networkx(graph)
    ondisk = OnDiskTransductiveDataset(
        graph_data=data,
        data_dir=index_dir,
        max_structure_size=3,
        force_rebuild=True,
    )
    ondisk.build_index()
    return ondisk
```

**Key Difference:**
- In-Memory: `triangles_dict` grows linearly → O(T) memory where T = triangle count
- OnDisk: Writes to disk, frees memory → O(1) constant

---

### **Phase 3: Progressive Scale Testing** (60 minutes)

**Test on progressively larger graphs until in-memory fails:**

**Test Suite:**
```python
GRAPH_CONFIGS = [
    {"name": "10K",  "nodes": 10000, "degree": 20},  # Baseline (both should work)
    {"name": "20K",  "nodes": 20000, "degree": 25},  # In-memory struggles
    {"name": "30K",  "nodes": 30000, "degree": 30},  # In-memory critical
    {"name": "40K",  "nodes": 40000, "degree": 30},  # In-memory likely fails
    {"name": "50K",  "nodes": 50000, "degree": 30},  # In-memory definitely fails
]
```

**For each graph:**
1. Generate synthetic graph (Watts-Strogatz)
2. Measure memory before
3. Run in-memory with storage (monitor memory)
4. Run OnDisk with optimized SQLite (monitor memory)
5. Compare results
6. **Stop when in-memory fails** ← This is the proof!

**Expected Results:**
```
10K:  In-Memory: ~400MB   OnDisk: ~100MB  → OnDisk better
20K:  In-Memory: ~1.5GB   OnDisk: ~200MB  → OnDisk better
30K:  In-Memory: ~3.5GB   OnDisk: ~400MB  → OnDisk much better
40K:  In-Memory: OOM      OnDisk: ~600MB  → OnDisk ONLY ✅
50K:  In-Memory: OOM      OnDisk: ~800MB  → OnDisk ONLY ✅
```

**The "Death vs. Survival" Moment:** 
- Graph size where in-memory OOMs but OnDisk succeeds = **PROOF COMPLETE** ✅

---

## 📊 **VALIDATION OUTPUTS REQUIRED**

### 1. Memory Profile Comparison Table
```
| Scale | In-Memory | OnDisk | Winner | Evidence |
|-------|-----------|--------|--------|----------|
| 10K   | 400MB     | 100MB  | OnDisk | Both work |
| 20K   | 1.5GB     | 200MB  | OnDisk | Both work |
| 30K   | 3.5GB     | 400MB  | OnDisk | Both work |
| 40K   | OOM       | 600MB  | OnDisk | OnDisk ONLY ✅ |
```

### 2. Time Comparison
```
| Scale | In-Memory Time | OnDisk Time | Triangles Found |
|-------|----------------|-------------|-----------------|
| 10K   | 1.2s          | 10.5s       | 325,922         |
| 20K   | 4.8s          | 38.2s       | 960,366         |
| 30K   | 12.5s         | 87.5s       | 2,295,833       |
| 40K   | OOM (crashed) | 156.3s      | 4,125,891       |
```

### 3. Correctness Validation
```
At each scale where both work:
- Triangle count match: ±0.1% acceptable
- 100% match on small subset queries
```

### 4. Final Validation Report

**Create:** `LARGE_SCALE_VALIDATION_FINAL.md`

**Must include:**
- Table showing memory usage at each scale
- Graph/chart showing linear growth (in-memory) vs constant (OnDisk)
- Clear identification of failure point
- Time measurements
- Correctness validation
- **Conclusion:** OnDisk enables scales impossible with in-memory

---

## ⚠️ **CRITICAL CONSTRAINTS**

### DO:
- ✅ Use synthetic graphs (Watts-Strogatz) - controllable and reproducible
- ✅ Monitor memory continuously during execution
- ✅ Store data in both approaches (fair comparison)
- ✅ Stop testing when in-memory fails (that's the proof!)
- ✅ Verify all existing tests still pass (65 tests)

### DON'T:
- ❌ Artificially limit RAM (let dataset be natural bottleneck)
- ❌ Run git commands
- ❌ Break existing tests
- ❌ Compare counting vs storing (unfair)
- ❌ Skip SQLite optimization (critical for fair memory usage)

---

## 🎯 **SPECIFIC OGBN-PRODUCTS GUIDANCE**

**Note:** Previous attempts to use ogbn-products failed because OGB library loads everything into memory before we can process it.

**IF you want to try ogbn-products:**

**Option A: Skip It** (Recommended)
- Synthetic graphs provide clearer proof
- We already have "attempted ogbn-products → OOM" as necessity proof
- Focus on 40K-50K synthetic graphs

**Option B: Try with Memory Mapping** (Advanced)
- Use `numpy.load(mmap_mode='r')` to access raw data
- Build graph structure incrementally
- Likely still won't work (OGB's internal loading)
- **Only attempt if Phase 3 completes successfully first**

---

## 📋 **EXECUTION CHECKLIST**

### Step 1: SQLite Optimization (30 min)
- [ ] Modify `topobench/data/index.py`
- [ ] Add PRAGMA statements
- [ ] Add batch commits
- [ ] Test on 10K graph
- [ ] Verify memory drops

### Step 2: Fair Comparison (30 min)
- [ ] Create `prove_ondisk_fair_comparison.py`
- [ ] Implement in-memory with storage
- [ ] Implement OnDisk with optimizations
- [ ] Test on 10K graph
- [ ] Verify correctness match

### Step 3: Progressive Testing (60 min)
- [ ] Test 10K (both should work)
- [ ] Test 20K (both should work, OnDisk better)
- [ ] Test 30K (in-memory struggles)
- [ ] Test 40K (in-memory likely fails) ← TARGET
- [ ] Test 50K if 40K didn't fail
- [ ] Document failure point

### Step 4: Documentation (30 min)
- [ ] Create `LARGE_SCALE_VALIDATION_FINAL.md`
- [ ] Include memory comparison table
- [ ] Include time measurements
- [ ] Include correctness validation
- [ ] **Document "Death vs. Survival" point**

**Total Time:** ~2.5 hours

---

## 🎊 **SUCCESS CRITERIA**

You have successfully completed this task when:

1. ✅ SQLite memory overhead reduced (<100MB for 10K graph)
2. ✅ Fair comparison implemented (both store triangles)
3. ✅ Found graph size where in-memory fails but OnDisk succeeds
4. ✅ Documented with measurements and tables
5. ✅ All 65 existing tests still pass
6. ✅ Created `LARGE_SCALE_VALIDATION_FINAL.md` report

**The "Death vs. Survival" proof:** 
- Must show at least ONE graph size (30K-50K nodes) where:
  - In-Memory: ❌ OOM or >3GB memory
  - OnDisk: ✅ Success with <1.5GB memory
  - **This is the proof needed for submission!**

---

## 📞 **REFERENCE FILES**

**Read First:**
- `HANDOFF_TO_NEXT_AGENT.md` - Complete context
- `OPTIMIZATION_DETAILS.md` - Technical background

**Modify:**
- `topobench/data/index.py` - Add SQLite optimizations

**Create:**
- `prove_ondisk_fair_comparison.py` - Fair comparison script
- `LARGE_SCALE_VALIDATION_FINAL.md` - Final report

**Test:**
- Run: `pytest test/data/preprocessor/test_ondisk_inductive.py -v`
- Expected: 21 passed ✅

---

## 💡 **TIPS FOR SUCCESS**

1. **Start small:** Optimize SQLite on 10K first, verify it works
2. **Fair is fair:** In-memory must store data, not just count
3. **Progressive testing:** Don't jump to 50K, test 10K→20K→30K→40K
4. **Stop at failure:** When in-memory OOMs, **that's your proof!** No need to go larger
5. **Document everything:** Tables, memory plots, timing - make it visual
6. **Be honest:** If OnDisk still uses more memory after optimization, document why

---

**You have all the context, tools, and plan. Execute the validation and prove OnDisk works at scale!** 🚀

**Expected outcome:** Clear proof that OnDisk enables 40K-50K node graphs where in-memory fails. This completes the validation for submission.
