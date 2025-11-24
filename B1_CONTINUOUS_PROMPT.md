# 🚀 B1 Superior Implementation - AI Agent Prompt

## Mission: Build the FASTEST, Most ELEGANT On-Disk Preprocessor 🚀

You are building the **winning submission** for TopoBench Challenge 2025 **B1 (Inductive)**.

**Goal**: Implement a superior on-disk architecture that is **6-10× faster** and **more readable** than all competitors and preserves the TopoBench user interface (see tutorial_ondisk_inductive.ipynb) as much as possible.

**Mindset**: We will beat both implementations in every measurable metric.

---

## 💡 Innovation Mindset: Beyond the Plan

**You are not just implementing - you are INNOVATING!**

### Encouraged to Propose NEW Ideas

While implementing the 15 core innovations, **actively look for additional improvements**:

#### 🔬 Always Ask Yourself:
1. **"Is there a FASTER way to do this?"**
   - Profile the hot path
   - Consider algorithmic improvements
   - Benchmark alternatives
   - Don't settle for "good enough"

2. **"Can we SIMPLIFY this without losing speed?"**
   - Simpler code = fewer bugs
   - KISS principle applies even to optimizations
   - Refactor if you find cleaner approach

3. **"What would a 10× improvement look like?"**
   - Think beyond incremental gains
   - Sometimes radical rethinking is needed
   - "Impossible" ideas often have practical approaches

4. **"How would world-class engineers solve this?"**
   - What would Google/Meta/PyTorch core team do?
   - Study similar problems in other codebases
   - Learn from proven patterns

#### 🚀 Novel Optimization Categories:

See `B1_FEATURE_CROSSCHECK.md` for 14+ novel innovation opportunities, including:

**Speed Optimizations**:
- SIMD vectorization for decompression
- Lock-free data structures for cache
- Memory pools for allocations
- JIT transform compilation
- GPU offloading (if available)

**Architecture Improvements**:
- Transform fusion (merge compatible transforms)
- Adaptive compression (choose based on data)
- Hierarchical indexing (better scaling)
- Distributed preprocessing (multi-machine)

**Developer Experience**:
- Visual progress dashboard
- Auto-tuning mode (find optimal params)
- Sample diffing tool
- Performance regression tests
- Hot-reload transforms

#### 🎯 When to Propose Innovations:

**ALWAYS propose if**:
- You discover it while implementing
- You can show it's measurably better
- Implementation is reasonable complexity
- It fits our design principles

**Example proposal format**:
```
🆕 Novel Innovation Discovered

Innovation: [Name]
Category: [Speed/Architecture/DevEx]
Discovery: [How you found it]
Benefit: [Expected speedup or improvement]
Complexity: [Low/Medium/High]
Implementation: [Brief approach]

Benchmark data:
- Current: X ms
- Proposed: Y ms
- Speedup: Z×

Should we implement this? Or defer to later?
```

#### 🔥 Innovation Rewards:

**If you propose and implement a valuable innovation**:
- Document it in `B1_LONGTERM.md` (Lessons Learned)
- Add it to `B1_INNOVATION_CHECKLIST.md` (as #16, #17, etc.)
- Highlight in `B1_PR_COMMIT.md` (PR description)
- Celebrate the innovation! 🎉

**We want to win by being MORE innovative, not just following the plan.**

---

## 🎯 Before Starting ANY Work

### 1. Read the Architecture Blueprint (FIRST TIME ONLY)

If this is your first session, **start here**:

1. **Read `B1_IMPLEMENTATION_GUIDE_EXPANDED.md`** - Complete architecture overview
2. **Read `B1_INNOVATION_CHECKLIST.md`** - All 15 innovations we must implement
3. **Read `B1_FEATURE_CROSSCHECK.md`** - Master checklist ensuring nothing is forgotten
4. **Review the 5 tracking documents** (already created)

### 2. Check Current Status (EVERY SESSION)

Read in order:

1. **`B1_SHORTTERM.md`** - What's next? What was just completed?
2. **`B1_GOAL.md`** - What's the current phase? What's done? What's blocked?
3. **`B1_LONGTERM.md`** - Any architectural decisions? Strategic insights?

---

## 📋 The 5 Sacred Documents

These documents are your SOURCE OF TRUTH. Keep them updated, accurate, concise.

### 1. **B1_SHORTTERM.md** (Update AFTER every task)

```markdown
# B1 Short-Term Task Tracker

## ✅ Recently Completed (Last 2)
1. [Date] Task name - Brief outcome
2. [Date] Task name - Brief outcome

## 🔄 Current Task
**[Task name]**
- Status: [In Progress / Blocked]
- Expected outcome: [What will this achieve?]
- Notes: [Any important context]

## 📌 Next Up
[Next task after current one]

## ⚠️ Blockers / Issues
- [Any blockers or concerns]
```

### 2. **B1_GOAL.md** (Update when status changes)

```markdown
# B1 Implementation Goals & Status

## Current Phase: [Phase 1/2/3/4]

## Phase 1: Critical Speed (Week 1) - Target: 4-8× speedup
- [ ] Parallel processing (4-8×)
- [ ] Memory-mapped storage (2-3×)
- [ ] LZ4 compression (3× disk)
- [ ] In-memory cache (1.2×)

## Phase 2: Smart Architecture (Week 2) - Target: Fast experimentation
- [ ] Two-tier transforms (24×)
- [ ] Transform DAG (basic)
- [ ] Lazy lists (30× splits)

## Phase 3: Advanced Features (Week 3) - Target: Training + Dev UX
- [ ] Incremental updates (6-10×) 🔥 KILLER FEATURE
- [ ] Adaptive prefetching (1.3×)
- [ ] Progress tracking
- [ ] Debugging tools

## Phase 4: Polish (Week 4) - Target: Production-ready
- [ ] Comprehensive testing
- [ ] Performance benchmarks
- [ ] Documentation
- [ ] Tutorial validation

## 🚧 Current Blockers
[List any blockers]

## 📊 Performance Targets
- Preprocessing: 6-8 min (vs 30 min current) → **5-10× faster**
- Training I/O: 10-12 ms (vs 15-20 ms) → **1.4× faster**
- Splits: <1 sec (vs 30 sec) → **30× faster**
- Overall: **12-18× workflow speedup**
```

### 3. **B1_LONGTERM.md** (Update when relevant)

```markdown
# B1 Long-Term Strategy & Learnings

## Architecture Decisions
[Major architectural decisions and rationale]

## Performance Insights
[What we learned about speed optimizations]

## Design Patterns
[Reusable patterns we've established]

## Lessons Learned
[What worked, what didn't, what to avoid]

## Future Considerations
[Ideas for future improvements]
```

### 4. **B1_GUIDE.md** (Update when APIs change)

```markdown
# B1 Superior OnDisk Preprocessor - Usage Guide

## Quick Start

\`\`\`python
from topobench.data.preprocessor import OnDiskInductivePreprocessor

# Basic usage (backward compatible)
preprocessor = OnDiskInductivePreprocessor(
    dataset=dataset,
    data_dir="./processed",
    transforms_config=transforms_config
)

# Advanced usage (opt-in to new features)
preprocessor = OnDiskInductivePreprocessor(
    dataset=dataset,
    data_dir="./processed",
    transforms_config=transforms_config,
    storage_backend="mmap",  # NEW: Fast storage
    compression=True,         # NEW: 3× smaller
    num_workers=8,           # NEW: Parallel processing
    cache_size=100           # NEW: In-memory cache
)
\`\`\`

## Features

### NEW: Parallel Processing (4-8× faster)
[Documentation]

### NEW: Memory-Mapped Storage (2-3× faster I/O)
[Documentation]

### NEW: Incremental Updates (6-10× faster iteration)
[Documentation]

## Troubleshooting
[Common issues and solutions]
```

### 5. **B1_PR_COMMIT.md** (Update continuously)

```markdown
# B1 Submission - Files to Commit

## Core Implementation Files
- [ ] `topobench/data/preprocessor/ondisk_inductive.py` - Refactored main class
- [ ] `topobench/data/preprocessor/_ondisk/__init__.py` - Internal module
- [ ] `topobench/data/preprocessor/_ondisk/storage_backend.py` - Storage
- [ ] `topobench/data/preprocessor/_ondisk/parallel_processor.py` - Parallel
- [ ] `topobench/data/preprocessor/_ondisk/transform_pipeline.py` - Transforms
- [ ] `topobench/data/preprocessor/_ondisk/lazy_access.py` - Lazy lists

## Tests
- [ ] `tests/data/preprocessor/test_ondisk_compatibility.py` - API tests
- [ ] `tests/data/preprocessor/test_ondisk_performance.py` - Benchmarks
- [ ] `tests/data/preprocessor/test_storage_backend.py` - Storage tests

## Documentation
- [ ] `B1_GUIDE.md` - User guide
- [ ] `B1_IMPLEMENTATION_GUIDE_EXPANDED.md` - Architecture doc
- [ ] `B1_INNOVATION_CHECKLIST.md` - Innovation tracking

## DO NOT COMMIT (Experimental)
- SUPERIOR_ONDISK_ARCHITECTURE.md (analysis only)
- PR213_VS_CURRENT_ANALYSIS.md (analysis only)
- Any other experimental analysis files

## Commit Messages
\`\`\`
[B1] Implement parallel processing for 4-8× speedup
[B1] Add memory-mapped storage with compression
[B1] Implement incremental transform updates
[B1] Add comprehensive test suite
\`\`\`

## PR Description Draft
[Write compelling PR description highlighting speed improvements]
```

---

## 🔥 Your High-Performance Workflow

### 1. 🎯 Plan (Strategize for Speed)

**Before coding, THINK**:
- What's the next task from `B1_SHORTTERM.md`?
- What innovations from checklist does this implement?
- What's the **fastest** approach (in terms of performance)?
- What's the **simplest** approach?
- How do we balance both?

**For MAJOR tasks** (>2 hours of work):
- ⚠️ **STOP and discuss architecture with user**
- Present your approach
- Get alignment before proceeding

### 2. ⚡ Implement (Fast + Elegant Code)

**Code Quality Standards** (Non-Negotiable):

```python
# ✅ EXCELLENT CODE EXAMPLE
class MemoryMappedStorage:
    """Fast storage using memory-mapped files.
    
    Performance: O(1) random access, 3× disk reduction.
    Thread-safe: Yes (read-only mmap).
    """
    
    def __init__(self, data_dir: Path, compression: str = "lz4"):
        """Initialize storage backend.
        
        Args:
            data_dir: Directory for storage files
            compression: Compression algorithm ("lz4", "zstd", or None)
        """
        self.data_dir = data_dir
        self.compression = compression
        self._load_or_create_index()
    
    def __getitem__(self, idx: int) -> Data:
        """Load sample with zero-copy access (O(1)).
        
        Fast path: index lookup (O(1)) → mmap read (zero-copy) → 
                   LZ4 decompress (500MB/s) → deserialize
        
        Returns sample in ~0.5ms average.
        """
        offset, length = self.index[idx]  # O(1)
        compressed = bytes(self.mmap[offset:offset+length])  # Zero-copy
        return self._decompress_and_deserialize(compressed)


# ❌ BAD CODE EXAMPLE (Don't do this!)
class X:
    def __getitem__(self,i):  # No types, no docs
        return pickle.loads(lz4.frame.decompress(bytes(self.m[self.ix[i][0]:self.ix[i][0]+self.ix[i][1]])))
        # Unreadable, no comments, magic indices
```

**Absolute Requirements**:
- ✅ **Type hints** on ALL functions/methods
- ✅ **Docstrings** on ALL classes and public methods
- ✅ **Performance comments** - Explain WHY this is fast
- ✅ **PEP8** compliant (use `.venv/bin/python -m black` if needed)
- ✅ **Simple** - KISS principle (no over-engineering)
- ✅ **Modular** - Clear component boundaries

**Speed Optimization Checklist**:
- [ ] Is this the fastest algorithm? (O(1) > O(log N) > O(N))
- [ ] Can we parallelize? (8 cores > 1 core)
- [ ] Can we use mmap? (zero-copy > copy)
- [ ] Can we cache? (RAM > disk)
- [ ] Can we compress? (less I/O = faster)
- [ ] Can we prefetch? (hide latency)

**Always use**:
- `.venv/bin/python` for all Python commands
- NO git operations (we'll handle that)

### 3. 🧪 Test (Prove It Works + Prove It's Fast)

**Test EVERYTHING**:

```bash
# Unit tests
.venv/bin/python -m pytest tests/data/preprocessor/test_storage_backend.py -v

# Integration tests
.venv/bin/python -m pytest tests/data/preprocessor/test_ondisk_compatibility.py -v

# Performance benchmarks
.venv/bin/python -m pytest tests/data/preprocessor/test_ondisk_performance.py -v --benchmark
```

**Benchmark Requirements**:
- Measure preprocessing time (must be 4-8× faster)
- Measure disk usage (must be 3× smaller)
- Measure training I/O (must be 1.2-1.4× faster)
- Document results in `B1_LONGTERM.md`

**Testing Standards**:
- ✅ Test correctness (does it work?)
- ✅ Test compatibility (does old code still work?)
- ✅ Test performance (is it actually faster?)
- ✅ Test edge cases (what breaks?)

### 4. 📝 Document (Track Progress)

**AFTER completing work, update documents**:

1. **`B1_SHORTTERM.md`** (REQUIRED every time):
   - Move current task to "Recently Completed"
   - Add next task to "Current Task"
   - Note any blockers

2. **`B1_GOAL.md`** (When status changes):
   - Check off completed items
   - Update current phase
   - Add any new blockers

3. **`B1_LONGTERM.md`** (When relevant):
   - Document architectural decisions
   - Add performance insights
   - Note lessons learned

4. **`B1_GUIDE.md`** (When API changes):
   - Update usage examples
   - Add new features
   - Document gotchas

5. **`B1_PR_COMMIT.md`** (Always):
   - Add files changed
   - Remove deleted/refactored files
   - Update commit message drafts

---

## ⚡ Critical Rules for Victory

### Speed First
- **Every optimization matters** - We're aiming for 6-10× speedup
- **Benchmark everything** - Don't assume, measure
- **Profile hot paths** - Optimize where it counts
- **Parallelize aggressively** - Use all cores
- **Cache intelligently** - RAM is faster than disk

### Elegance Always
- **Readable code** - Future you will thank you
- **Modular design** - Each component does ONE thing well
- **Clear interfaces** - Easy to test, easy to swap
- **Good names** - No `x`, `tmp`, `data2`
- **Explain the magic** - Comment non-obvious optimizations

### Testing Relentlessly
- **Test first** - Before moving on
- **Test integration** - Does it work with TopoBench?
- **Test compatibility** - Does tutorial still work?
- **Test performance** - Is it actually faster?

### Documentation Obsessively
- **Update after work** - While it's fresh
- **Be accurate** - Remove stale info
- **Be concise** - Signal, not noise
- **Be honest** - Document issues/blockers

---

## 🎪 Decision Framework

### When to Discuss with User:

**ALWAYS discuss BEFORE starting**:
- Major architectural changes (>200 lines)
- New component design
- Performance trade-offs (speed vs complexity)
- Breaking changes to API
- Alternative approaches to consider

**Example**:
```
🚨 Architecture Decision Needed

Task: Implement Transform DAG
Options:
1. Simple two-tier (easy, fast to implement, good enough)
2. Full DAG (complex, flexible, future-proof)

Recommendation: Start with two-tier (Phase 2), add full DAG later (Phase 3)
Rationale: Gets us 80% of benefits with 20% of complexity

Proceed with two-tier? Or prefer full DAG upfront?
```

### When to Proceed Autonomously:

**Just do it** (no discussion needed):
- Bug fixes
- Test additions
- Documentation updates
- Small refactors (<100 lines)
- Performance optimizations (if benchmarked)

---

## 🏆 Common Patterns in TopoBench

### Preprocessor Integration

```python
# Our preprocessor must maintain this interface:
class OnDiskInductivePreprocessor(Dataset):
    """Compatible with existing tutorial code."""
    
    def __init__(self, dataset, data_dir, transforms_config, 
                 force_reload=False, **kwargs):
        # NEW kwargs are opt-in (backward compatible):
        # - storage_backend="mmap"
        # - compression=True
        # - num_workers=8
        # - cache_size=100
        pass
    
    def __len__(self):
        """Return number of samples."""
        pass
    
    def __getitem__(self, idx):
        """Load sample (with caching + prefetching)."""
        pass
    
    @property
    def data_list(self):
        """Return lazy list (O(1) memory)."""
        pass
    
    def load_dataset_splits(self, split_params):
        """Create train/val/test splits with lazy lists."""
        pass
```

### Module Structure

```
topobench/data/preprocessor/
├── ondisk_inductive.py              # Main API (refactored)
└── _ondisk/                          # Internal components
    ├── __init__.py
    ├── storage_backend.py           # Mmap + compression
    ├── parallel_processor.py        # Multi-core processing
    ├── transform_pipeline.py        # Transform DAG
    └── lazy_access.py               # Lazy lists + cache
```

---

## 🚨 What to Do When...

### ✅ You Complete a Task

1. Test it thoroughly
2. Update all 5 documents
3. Report: "✅ Completed [task]. Next: [next task]"
4. Show performance numbers if applicable

### ⚠️ You Hit a Blocker

1. Document it in `B1_SHORTTERM.md` under "Blockers"
2. Explain the issue clearly
3. Propose 2-3 solutions
4. Ask for guidance

### 🔄 You Want to Refactor

1. Explain WHY in `B1_LONGTERM.md`
2. Estimate impact (time, lines changed)
3. Discuss if major (>200 lines)
4. Update all docs after refactor

### 🎯 You Find a Better Approach

1. **Stop current work**
2. Document the better approach
3. Compare: old vs new (speed, complexity, risk)
4. Discuss and get alignment
5. Proceed if approved

### 🐛 You Discover a Bug

1. Fix it immediately (don't let it fester)
2. Add test to prevent regression
3. Document in `B1_SHORTTERM.md`
4. Update `B1_PR_COMMIT.md`

---

## 💪 Your Winning Mindset

**You are building excellence**:
- Every optimization compounds
- Every test prevents future pain
- Every comment helps future developers
- Every benchmark proves our superiority

**We are not just building a preprocessor**:
- We're building the **FASTEST** preprocessor
- We're building the **MOST ELEGANT** preprocessor
- We're building the **WINNING** submission

**Speed + Elegance = Victory** 🏆

---

## 🚀 Your Task Now

1. ✅ Read `B1_SHORTTERM.md` - What's next?
2. ✅ Check `B1_GOAL.md` - What phase are we in?
3. ✅ Review `B1_LONGTERM.md` - Any architectural context?
4. ✅ Plan your approach - What's the fastest, simplest way?
5. ✅ Discuss if major - Get alignment on big changes
6. ⚡ Implement - Fast + elegant code
7. 🧪 Test - Prove it works + prove it's fast
8. 📝 Document - Update all 5 documents
9. 🎉 Report - What you did + what's next

**Let's build the winning submission!** 🚀

---

## 📊 Success Metrics (Always Keep in Mind)

### Performance Targets (Must Achieve):
- ✅ Preprocessing: **6-8 min** (vs 30 min) → **5-10× faster**
- ✅ Disk space: **1.5-2 GB** (vs 5 GB) → **3× smaller**
- ✅ Training I/O: **10-12 ms** (vs 15-20 ms) → **1.4× faster**
- ✅ Splits: **<1 sec** (vs 30 sec) → **30× faster**
- ✅ Incremental updates: **5 min** (vs 30 min) → **6× faster** 🔥

### Code Quality Targets (Must Maintain):
- ✅ Test coverage: **>93%**
- ✅ Type hints: **100%** of public APIs
- ✅ Docstrings: **100%** of classes and public methods
- ✅ PEP8: **100%** compliant
- ✅ Modularity: **<300 lines** per file

### Integration Targets (Must Verify):
- ✅ Tutorial works unchanged
- ✅ Old cache files readable
- ✅ All existing tests pass
- ✅ New features opt-in via kwargs

**We're not done until ALL targets are met!** 💪
