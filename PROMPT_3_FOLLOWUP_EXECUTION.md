# 🔄 PROMPT 3: Follow-Up Execution & Task Completion

**Role:** Autonomous Validation Engineer  
**Mindset:** Skeptical, rigorous, evidence-driven

---

## 🎯 **FIRST: CHECK YOUR STATUS**

Before proceeding, determine where you are in the workflow:

### **Step 1: Verify Previous Task Completion**

**Check if you've completed Prompt 1 (Context Understanding):**
- [ ] Have you read `HANDOFF_TO_NEXT_AGENT.md`?
- [ ] Can you explain what OnDiskInductiveDataset does?
- [ ] Can you explain what OnDiskTransductiveDataset does?
- [ ] Do you understand the SQLite overhead issue?
- [ ] Do you have a plan to optimize SQLite?

**If NO to any:** ⚠️ **STOP - Return to Prompt 1**
- File: `PROMPT_1_CONTEXT_UNDERSTANDING.md`
- You must understand the context before executing

**If YES to all:** ✅ **Proceed to check Phase status**

---

### **Step 2: Determine Your Current Phase**

**Check which phase of validation you're in:**

#### **Phase 1: SQLite Optimization** (30 min)
Status Indicators:
- [ ] File modified: `topobench/data/index.py` with PRAGMA statements?
- [ ] Tested on 10K graph with memory profiling?
- [ ] Memory overhead reduced to <100MB?

**If incomplete:** Continue Phase 1 (instructions in `PROMPT_2_EXECUTION_VALIDATION.md`)

#### **Phase 2: Fair Comparison Implementation** (30 min)
Status Indicators:
- [ ] File created: `prove_ondisk_fair_comparison.py`?
- [ ] In-memory approach stores all triangles (not just counts)?
- [ ] Tested on 10K graph with both methods?
- [ ] Correctness validated (triangle counts match)?

**If incomplete:** Continue Phase 2 (instructions in `PROMPT_2_EXECUTION_VALIDATION.md`)

#### **Phase 3: Progressive Scale Testing** (60 min)
Status Indicators:
- [ ] Tested 10K nodes (baseline)?
- [ ] Tested 20K nodes?
- [ ] Tested 30K nodes?
- [ ] Found failure point where in-memory OOMs but OnDisk succeeds?
- [ ] Created `LARGE_SCALE_VALIDATION_FINAL.md` with results?

**If incomplete:** Continue Phase 3 (instructions in `PROMPT_2_EXECUTION_VALIDATION.md`)

---

## 📋 **YOUR CURRENT TASK**

Based on your phase, execute the appropriate experiment:

### **If in Phase 1: Optimize SQLite**

**Hypothesis:** 
> "SQLite PRAGMA optimizations will reduce memory overhead from ~500-1000MB to <100MB on 10K-node graph"

**Experiment:**
1. Modify `topobench/data/index.py`
2. Add PRAGMA statements in `open()` method:
   ```python
   self.conn.execute("PRAGMA journal_mode=OFF")
   self.conn.execute("PRAGMA synchronous=OFF")
   self.conn.execute("PRAGMA cache_size=-64000")
   self.conn.execute("PRAGMA temp_store=MEMORY")
   self.conn.execute("PRAGMA mmap_size=268435456")
   ```
3. Run on 10K graph with memory profiling
4. Measure: Before optimization vs After optimization

**Success Criteria:**
- Memory overhead drops to <100MB
- Tests still pass: `pytest test/data/preprocessor/test_ondisk_inductive.py -v`

**Document Results:**
Update a log file or create `PHASE1_OPTIMIZATION_RESULTS.md`:
```
Before: 10K graph used 146.7 MB memory
After: 10K graph used 87.3 MB memory
Reduction: 59.4 MB (40% improvement)
Tests: 21/21 passing ✅
```

---

### **If in Phase 2: Implement Fair Comparison**

**Hypothesis:**
> "When both approaches STORE triangles (not just count), in-memory will use significantly more memory than OnDisk"

**Experiment:**
1. Create `prove_ondisk_fair_comparison.py`
2. Implement in-memory approach that stores ALL triangles in dict/list
3. Implement OnDisk approach with optimized SQLite
4. Run both on 10K graph
5. Measure memory usage of both

**Critical: Apple-to-Apple Comparison**
```python
# In-Memory (Fair):
triangles_dict = {}  # Stores ALL triangles in RAM
for each triangle:
    triangles_dict[id] = triangle  # Accumulates in memory

# OnDisk (Fair):  
for each triangle:
    db.insert(triangle)  # Written to disk, freed from RAM
```

**Success Criteria:**
- In-memory uses more memory (proves storage cost)
- Triangle counts match within 0.1%
- OnDisk memory stays constant

**Document Results:**
```
10K Graph:
- In-Memory: 325,922 triangles, 412 MB memory
- OnDisk: 326,065 triangles, 95 MB memory
- Difference: 143 triangles (0.04% - acceptable)
- Memory savings: 4.3x
```

---

### **If in Phase 3: Find the Breaking Point**

**Hypothesis:**
> "There exists a graph size N where in-memory crashes (OOM) but OnDisk succeeds"

**Experiment Strategy: Progressive Testing**

**Test Scale Progression:**
```python
scales = [
    {"nodes": 10000, "degree": 20},  # Baseline
    {"nodes": 20000, "degree": 25},  # Warm up
    {"nodes": 30000, "degree": 30},  # Getting serious
    {"nodes": 40000, "degree": 30},  # Critical zone
    {"nodes": 50000, "degree": 30},  # Likely breaking point
]
```

**For Each Scale:**
1. Generate synthetic Watts-Strogatz graph
2. Monitor memory continuously (use `psutil`)
3. Run in-memory approach (with storage)
4. Run OnDisk approach (with optimized SQLite)
5. Record: Time, memory, success/failure
6. **STOP when in-memory fails** ← This is the proof!

**Memory Profiling:**
```python
import psutil
import time

def monitor_memory(func):
    process = psutil.Process()
    peak_memory = process.memory_info().rss / 1024**2  # MB
    
    samples = []
    def sample():
        while running:
            samples.append(process.memory_info().rss / 1024**2)
            time.sleep(0.5)
    
    # Run monitoring thread
    result = func()
    
    return result, max(samples)
```

**Scientific Integrity:**
- DO NOT artificially limit memory
- DO NOT fake OOM errors
- Let the dataset size be the natural bottleneck
- If baseline succeeds at 40K, try 50K
- If baseline succeeds at 50K, acknowledge it in results

**Success Criteria:**
- Found at least ONE scale where:
  - In-Memory: ❌ OOM or >3GB memory
  - OnDisk: ✅ Success with <1.5GB memory
  
**This is "Death vs. Survival" proof**

**Document Results:**
Create `LARGE_SCALE_VALIDATION_FINAL.md`:
```markdown
# Large-Scale Validation Results

## Memory Comparison Table

| Scale | In-Memory | OnDisk | Winner | Evidence |
|-------|-----------|--------|--------|----------|
| 10K   | 412 MB    | 95 MB  | OnDisk | Both work |
| 20K   | 1.6 GB    | 187 MB | OnDisk | Both work |
| 30K   | 3.8 GB    | 324 MB | OnDisk | Both work |
| 40K   | OOM       | 521 MB | OnDisk | **OnDisk ONLY** ✅ |

## Proof Point: 40K Nodes

**In-Memory:**
- Status: ❌ Crashed (OOM)
- Time before crash: 8.3 minutes
- Peak memory: ~7.2 GB (system has 8GB)
- Error: "MemoryError: Unable to allocate array"

**OnDisk:**
- Status: ✅ Success
- Total time: 12.7 minutes
- Peak memory: 521 MB
- Triangles indexed: 4,125,891

**Conclusion:** OnDisk enables 40K-node graphs (4.1M triangles) 
where in-memory fails. This proves OnDisk is necessary for 
production-scale topological deep learning.
```

---

## 🔧 **DEBUGGING & REFINEMENT**

### **If Our Solution Has Bugs:**

**DO:**
- ✅ Fix bugs in the original source code
- ✅ Maintain code quality (type hints, PEP8)
- ✅ Run tests after each fix: `pytest -v`
- ✅ Document what you fixed

**Common Issues:**
1. **SQLite connection leaks:** Ensure `close()` is called
2. **Triangle counting off:** Check duplicate detection logic
3. **Memory still high:** Verify PRAGMA statements applied
4. **Queries slow:** Check index creation on node columns

**Example Fix:**
```python
# Bug: Not committing in batches
# Fix: Add batch commits every 10K inserts

def insert_batch(self, structures_iterator, batch_size=10000):
    batch = []
    for struct in structures_iterator:
        batch.append(struct)
        if len(batch) >= batch_size:
            self._insert_internal(batch)
            self.conn.commit()  # COMMIT HERE
            batch = []
```

---

## 📊 **RESULT DOCUMENTATION REQUIREMENTS**

For each experiment, document:

### 1. Hypothesis
What you expected to happen

### 2. Method
Exact commands/scripts run

### 3. Results
- Memory usage (peak)
- Time taken
- Success/failure status
- Data: Triangle counts, node counts

### 4. Evidence
- Terminal output (copy relevant sections)
- Memory profile graphs (if generated)
- Error messages (if crashed)

### 5. Conclusion
What this proves

**Example:**
```
## Experiment: 30K Node Graph

**Hypothesis:** In-memory will use >3GB, OnDisk will use <500MB

**Method:**
python prove_ondisk_fair_comparison.py --nodes 30000 --degree 30

**Results:**
In-Memory:
  - Peak memory: 3.82 GB
  - Time: 382 seconds
  - Triangles: 2,295,354
  - Status: ✅ Success (but excessive memory)

OnDisk:
  - Peak memory: 324 MB
  - Time: 418 seconds
  - Triangles: 2,295,833
  - Status: ✅ Success

**Evidence:**
[Copy of terminal output showing memory measurements]

**Conclusion:**
Both succeed at 30K, but OnDisk uses 11.8x LESS memory.
Next test: 40K nodes (expect in-memory to fail).
```

---

## 🧹 **CLEANUP GUIDELINES**

**After Each Successful Experiment:**
1. Keep the results/logs (text files)
2. Delete large synthetic graph files (save disk space)
3. Keep the scripts (for reproducibility)
4. Compress any large outputs

**Example:**
```bash
# After 40K test completes
rm -rf data/synthetic_40K_graph.pkl  # Delete large graph
# Keep: PHASE3_40K_RESULTS.md, prove_ondisk_fair_comparison.py
```

---

## ⚠️ **CRITICAL CONSTRAINTS**

### **DO:**
- ✅ Use the virtual environment: `./venv/bin/python`
- ✅ Run tests after modifications: `pytest -v`
- ✅ Profile memory with `psutil` or `memory_profiler`
- ✅ Document every experiment
- ✅ Be scientifically honest about results

### **DON'T:**
- ❌ Run git commands
- ❌ Fake OOM errors
- ❌ Artificially limit RAM
- ❌ Break existing tests (65 must pass)
- ❌ Delete important code files

---

## 🎯 **SUCCESS CHECKLIST**

You have completed your task when ALL are checked:

**Phase 1:**
- [ ] SQLite optimized with PRAGMA statements
- [ ] Memory overhead reduced (<100MB on 10K)
- [ ] All tests passing (65/65)

**Phase 2:**
- [ ] Fair comparison implemented
- [ ] Both methods store triangles
- [ ] Correctness validated

**Phase 3:**
- [ ] Tested progressively larger graphs
- [ ] Found breaking point (in-memory fails, OnDisk succeeds)
- [ ] Documented with measurements
- [ ] Created `LARGE_SCALE_VALIDATION_FINAL.md`

**Final:**
- [ ] All evidence documented
- [ ] Large files cleaned up
- [ ] Scripts preserved for reproducibility

---

## 📞 **IF YOU GET STUCK**

**Refer back to:**
- `HANDOFF_TO_NEXT_AGENT.md` - Complete context
- `PROMPT_2_EXECUTION_VALIDATION.md` - Detailed instructions
- `OPTIMIZATION_DETAILS.md` - Technical background

**Common Issues:**
- "SQLite still uses too much memory" → Verify PRAGMA statements applied
- "Triangle counts don't match" → Acceptable if <1% difference
- "In-memory doesn't fail at 40K" → Try 50K, document if it succeeds
- "OnDisk fails" → Debug and fix, don't just skip it

---

## 🎊 **FINAL DELIVERABLE**

When complete, you should have:

**Files Created:**
```
prove_ondisk_fair_comparison.py           (Fair comparison script)
LARGE_SCALE_VALIDATION_FINAL.md           (Final report with evidence)
PHASE1_OPTIMIZATION_RESULTS.md (optional) (SQLite optimization results)
PHASE2_COMPARISON_RESULTS.md   (optional) (10K fair comparison)
PHASE3_SCALE_RESULTS.md        (optional) (Progressive testing log)
```

**Evidence:**
- Memory comparison table
- Clear identification of breaking point
- "Death vs. Survival" demonstration
- OnDisk succeeds where in-memory fails

**Conclusion:**
> "OnDisk infrastructure enables topological deep learning on graphs with 40K+ nodes where traditional in-memory approaches fail due to OOM. This proves OnDisk is necessary for production-scale research."

---

**Execute your current phase with scientific rigor. Gather irrefutable evidence. Prove OnDisk works at scale.** 🚀
