# Git Commit Plan - Uncommitted Changes

**Date:** November 26, 2024  
**Status:** Organized into logical bundles

---

## 🎯 Critical Changes (Must Commit)

### **Bundle 1: Core Bug Fixes** 🐛
**Priority:** HIGH - Critical bug fixes

**Files:**
- `topobench/data/preprocessor/ondisk_inductive.py` (DAG cache collision fix)
- `topobench/data/datasets/_lazy.py` (pickling fix)
- `test/data/preprocessor/test_dag_caching.py` (new test)

**Commit Message:**
```
🐛 Fix DAG cache collisions and LazySubset pickling

- Fix cache collision for duplicate transforms with transform_id + hash
- Add comprehensive inline documentation (20+ lines) explaining fix
- Fix LazySubset/LazyDataloadDataset pickling by making indices a property
- Add test_dag_handles_duplicate_transforms_correctly to verify fix
```

**Why:** These are critical bug fixes that prevent incorrect caching behavior

---

### **Bundle 2: Parallel Merge Performance** ⚡
**Priority:** HIGH - Major performance improvement

**Files:**
- `topobench/data/preprocessor/ondisk_inductive.py` (parallel merge implementation)
- `test/data/preprocessor/test_parallel_merge.py` (new test)

**Commit Message:**
```
⚡ Add parallel shard merging for mmap backend

- Implement _write_shard_to_offset() for parallel-safe writes
- Update _merge_shards() to use parallel workers (10× faster merge)
- Add comprehensive timing instrumentation for bottleneck analysis
- Note: Compression remains bottleneck (~45% of time), not merge
```

**Why:** 10× faster merge phase, though overall speedup limited by compression

---

### **Bundle 3: Test Coverage Improvements** ✅
**Priority:** MEDIUM - Ensures robustness

**Files:**
- `test/data/preprocessor/test_ondisk_inductive.py` (additional tests)
- `topobench/data/utils/split_utils.py` (minor fix if any)

**Commit Message:**
```
✅ Expand test coverage for on-disk preprocessing

- Add comprehensive tests for edge cases
- Verify DAG caching behavior across scenarios
- Ensure parallel processing correctness
```

**Why:** Comprehensive testing prevents regressions

---

### **Bundle 4: Documentation and Tutorials** 📚
**Priority:** HIGH - User-facing improvements

**Files:**
- `tutorials/tutorial_ondisk_inductive_part1_getting_started.ipynb` (new)
- `tutorials/tutorial_ondisk_inductive_part2_advanced.ipynb` (new)
- `tutorials/tutorial_ondisk_inductive_final.ipynb` (updated with navigation)
- `README_DAG_CACHING.md` (new)
- `SPEED_VS_COMPRESSION_TRADEOFF.md` (new)

**Commit Message:**
```
📚 Add comprehensive on-disk preprocessing tutorials and docs

- Add Part 1: Getting Started (15-20 min tutorial)
- Add Part 2: Advanced Techniques (DAG caching, backends, parallel)
- Update main tutorial with navigation to both parts
- Add technical documentation (DAG caching, backend trade-offs)
- Professional tone with measured claims and honest trade-offs
```

**Why:** Critical for user adoption and understanding

---

### **Bundle 5: Benchmark Infrastructure** 📊
**Priority:** MEDIUM - For validation and PR

**Files:**
- `benchmarks/benchmark_comprehensive_pipeline.py` (new)
- `benchmarks/configs/test.yaml` (new)
- `benchmarks/configs/comprehensive.yaml` (new)

**Commit Message:**
```
📊 Add comprehensive benchmark suite for on-disk preprocessing

- Parallel speedup benchmark (worker scaling)
- DAG cache reuse benchmark (incremental speedup)
- Memory efficiency benchmarks (lifting + full training)
- Detailed config documentation with backend trade-offs
```

**Why:** Validates performance claims, useful for PR but may be removed

---

## 🗑️ Noise Files (Do NOT Commit)

### **Session/Planning Documents** (50+ files)
All these are temporary development artifacts:

- `*_SUMMARY.md` files (session logs)
- `*_GUIDE.md` files (planning docs)  
- `*_ANALYSIS.md` files (investigation notes)
- `*_EXPLANATION.md` files (technical deep-dives)
- `BUGFIX_*.md`, `IMPLEMENTATION_*.md`, `COMPLETION_*.md`
- `PR_COMMENT_*.md`, `PR_PREP_SUMMARY.md` (draft only)

**Action:** Add to `.gitignore` or delete

**Exceptions to keep:**
- `PR_COMMENT_DRAFT.md` - Useful for PR, but not in git
- `README_DAG_CACHING.md` - User-facing documentation ✅
- `SPEED_VS_COMPRESSION_TRADEOFF.md` - User-facing documentation ✅

---

### **Test/Diagnostic Scripts**
Temporary validation scripts:

- `diagnose_memory_issues.py`
- `test_ondisk_training_fix.py`

**Action:** Delete after validation complete

---

### **Benchmark Results**
Generated output directories:

- `benchmarks_test/`
- `results/`

**Action:** Add to `.gitignore` (keep configs, ignore outputs)

---

## 📋 Recommended Commit Order

**Commit in this sequence:**

1. **Bundle 1** 🐛 - Bug fixes first (critical)
2. **Bundle 2** ⚡ - Performance improvements (builds on fixes)
3. **Bundle 3** ✅ - Tests (validates fixes + improvements)
4. **Bundle 4** 📚 - Documentation (explains everything)
5. **Bundle 5** 📊 - Benchmarks (optional, for PR validation)

---

## 🧹 Cleanup Commands

### Delete Noise Files
```bash
# Delete all session/planning documents
rm -f *_SUMMARY.md *_GUIDE.md *_ANALYSIS.md *_EXPLANATION.md
rm -f BUGFIX_*.md IMPLEMENTATION_*.md COMPLETION_*.md
rm -f HANDOFF_*.md NOTEBOOK_UPDATE_GUIDE.md FILES_CREATED_SUMMARY.md
rm -f ISOLATED_*.md MEMORY_FULL_*.md ONDISK_TRAINING_BUG_*.md
rm -f PARALLEL_*.md PICKLING_*.md PUBLICATION_REPORT.md
rm -f READY_TO_BENCHMARK.md SCALE_*.md STREAMLINED_TESTS_SUMMARY.md
rm -f TEST_COVERAGE_*.md TEST_INTEGRATION_SUMMARY.md
rm -f VERIFICATION_SUMMARY.md VERIFY_PARALLEL_FIX.md
rm -f YOUR_REQUESTS_COMPLETED.md EXAMPLE_BENCHMARK_REPORT_WITH_NOTE.txt
rm -f BENCHMARK_*.md BOTTLENECK_INVESTIGATION.md CLEANUP_PLAN.md
rm -f COMPLETE_BENCHMARK_OUTPUTS_SUMMARY.md COMPREHENSIVE_FINAL_STATUS.md
rm -f DAG_*.md FINAL_*.md LAZY_IMPORT_CIRCULAR_DEPENDENCY_EXPLAINED.md
rm -f PR_COMMENT_PARALLEL_MERGE.md PR_PREP_SUMMARY.md
rm -f TUTORIAL_IMPLEMENTATION_SUMMARY.md TUTORIAL_RESTRUCTURE_PLAN.md
rm -f README_ONDISK_SECTION.md SESSION_FINAL_SUMMARY.md

# Keep these user-facing docs:
# - README_DAG_CACHING.md
# - SPEED_VS_COMPRESSION_TRADEOFF.md
# - PR_COMMENT_DRAFT.md (for PR, but don't commit)

# Delete test scripts
rm -f diagnose_memory_issues.py test_ondisk_training_fix.py
```

### Update .gitignore
```bash
# Add to .gitignore
echo "" >> .gitignore
echo "# Session/Development artifacts" >> .gitignore
echo "*_SUMMARY.md" >> .gitignore
echo "*_GUIDE.md" >> .gitignore
echo "*_ANALYSIS.md" >> .gitignore
echo "PR_COMMENT_*.md" >> .gitignore
echo "PR_PREP_SUMMARY.md" >> .gitignore
echo "" >> .gitignore
echo "# Benchmark outputs" >> .gitignore
echo "benchmarks_test/" >> .gitignore
echo "results/" >> .gitignore
echo "benchmark_tmp/" >> .gitignore
echo "" >> .gitignore
echo "# Diagnostic scripts" >> .gitignore
echo "diagnose_*.py" >> .gitignore
echo "test_*_fix.py" >> .gitignore
```

---

## 🎯 Final State After Commits

### Committed (Production Code)
- ✅ Bug fixes (cache collision, pickling)
- ✅ Performance improvements (parallel merge)
- ✅ Comprehensive tests
- ✅ User documentation (tutorials + technical docs)
- ✅ Benchmark suite (optional)

### Not Committed (Artifacts)
- ❌ 50+ session/planning documents
- ❌ Test/diagnostic scripts
- ❌ Benchmark output directories
- ❌ PR drafts (use externally)

### Clean Repository
- Clean git status
- Professional commits with gitmoji
- Clear history for reviewers

---

## 💡 Notes

**Why bundle commits?**
- Logical separation of concerns
- Easy to review
- Clear history
- Can cherry-pick if needed

**Why this order?**
- Fixes before features
- Tests after implementation
- Docs last (explains final state)
- Benchmarks optional (for PR validation)

**Gitmoji meanings:**
- 🐛 `:bug:` - Bug fixes
- ⚡ `:zap:` - Performance improvements
- ✅ `:white_check_mark:` - Tests
- 📚 `:books:` - Documentation
- 📊 `:bar_chart:` - Analytics/benchmarks

---

**Ready to commit!** 🚀
