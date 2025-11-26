# Cleanup Plan - Outdated Documentation Files

## Files to Remove (Outdated Session Documents)

### Benchmark Development Files (Superseded)
- `ACTUAL_PERFORMANCE_REPORT.md` - Old performance data
- `BENCHMARK_DELIVERABLES.md` - Superseded by current benchmarks
- `BENCHMARK_FIXES_SUMMARY.md` - Historical, fixed issues
- `BENCHMARK_FIX_NEEDED.md` - Issues resolved
- `BENCHMARK_GUIDE.md` - Outdated guide
- `BENCHMARK_IMPROVEMENTS_SUMMARY.md` - Historical
- `BENCHMARK_SUMMARY.md` - Old summary
- `BENCHMARK_VERIFICATION_RESULTS.md` - Old results
- `BENCHMARKING_WORKFLOW.md` - Outdated workflow
- `BENCHMARKS_FINAL_VERIFICATION.md` - Superseded
- `BENCHMARKS_TEST_RESULTS.md` - Old test results
- `BENCHMARKS_VERIFICATION_COMPLETE.md` - Historical
- `COMPREHENSIVE_BENCHMARKS_CREATED.md` - Historical
- `COMPREHENSIVE_PARALLEL_ANALYSIS.md` - Analysis done
- `CONFIGURABLE_BENCHMARKS_SUMMARY.md` - Historical
- `FINAL_BENCHMARKING_DELIVERABLES.md` - Historical
- `FINAL_BENCHMARK_SUCCESS_SUMMARY.md` - Historical
- `FINAL_BENCHMARK_SUMMARY.md` - Historical
- `FINAL_COMPLETE_VALIDATION_SUITE.md` - Historical
- `FINAL_ISOLATED_BENCHMARKING_SUMMARY.md` - Historical
- `FINAL_PARALLEL_IMPLEMENTATION_SUMMARY.md` - Historical
- `FINAL_VALIDATION_REPORT.md` - Historical
- `FULL_PIPELINE_RESULTS.md` - Old results

### DAG Caching Development Files (Complete)
- `DAG_CACHING_DECISION.md` - Historical decision doc
- `DAG_CACHING_FIX_PLAN.md` - Plan completed
- `DAG_CACHING_IMPLEMENTATION_COMPLETE.md` - Historical

### Bug Fix Development Files (Complete)
- `BUGFIX_OOM_INDEX_BUILDING.md` - Fixed
- `CACHE_BUG_FIX_SUMMARY.md` - Fixed
- `COMPREHENSIVE_BUG_FIXES.md` - Historical
- `CRITICAL_BUG_INVESTIGATION.md` - Resolved
- `INVESTIGATION_RESULTS.md` - Historical
- `MEMORY_BENCHMARK_EXPLANATION.md` - Outdated
- `MEMORY_BENCHMARK_FIX.md` - Fixed
- `MEMORY_BENCHMARK_FIX_SUMMARY.md` - Historical
- `MERGE_BOTTLENECK_ANALYSIS.md` - Resolved
- `MERGE_FIX_COMPLETE.md` - Historical
- `OPTION1_IMPLEMENTATION_COMPLETE.md` - Historical
- `OPTION3_DIRECT_MMAP_ANALYSIS.md` - Historical analysis
- `PARALLEL_BOTTLENECK_ANALYSIS.md` - Resolved

### General Development Files (Superseded)
- `CLEANUP_SUMMARY.md` - Old cleanup
- `COMPLETION_SUMMARY.md` - Historical
- `DISK_SPACE_AND_CLEANUP_ANALYSIS.md` - Old analysis
- `FILES_CREATED_SUMMARY.md` - Historical
- `FINAL_STATE_SUMMARY.md` - Old state
- `HANDOFF_TO_NEXT_AGENT.md` - Session handoff
- `IMPLEMENTATION_SUMMARY.md` - Historical
- `NOTEBOOK_UPDATE_GUIDE.md` - Outdated
- `OLD_FILES_TO_DELETE.md` - Historical cleanup list
- `OPTIMIZATION_SUMMARY.md` - Historical
- `PROGRESS_SUMMARY.md` - Old progress
- `VALIDATION_COMPREHENSIVE_REPORT.md` - Old validation

### Prompt/Planning Files (Keep for Reference but Review)
- `B1_BENCHMARK_PLAN.md` - Keep (active planning)
- `B1_CONTINUOUS_PROMPT.md` - Archive candidate
- `B1_DATASET_ARCHITECTURE.md` - Keep (architecture doc)
- `B1_FEATURE_CROSSCHECK.md` - Archive candidate
- `B1_GOAL.md` - Keep (goal definition)
- `B1_GUIDE.md` - Keep (implementation guide)
- `B1_IMPLEMENTATION_GUIDE_EXPANDED.md` - Archive candidate
- `B1_INNOVATION_CHECKLIST.md` - Archive candidate
- `B1_LONGTERM.md` - Keep (strategic planning)
- `B1_MISSION_FORWARD.md` - Archive candidate
- `B1_PR_COMMIT.md` - Archive candidate
- `B1_SHORTTERM.md` - Keep (tactical planning)
- `CONTINUOUS_PROMPT.md` - Archive candidate
- `GOAL.md` - Duplicate of B1_GOAL.md
- `GUIDE.md` - Duplicate of B1_GUIDE.md
- `INITIAL_PROMPT.md` - Archive candidate
- `LONGTERM.md` - Duplicate of B1_LONGTERM.md
- `SHORTTERM.md` - Duplicate of B1_SHORTTERM.md

### Current Active Documents (Keep)
- `ONDISK_ARCHITECTURE_ANALYSIS.md` - Important architecture doc
- `ONDISK_TRAINING_BUG_ANALYSIS.md` - Bug analysis (keep for reference)
- `ONDISK_TRAINING_BUG_FIX_SUMMARY.md` - Recent fix summary
- `OGBN_PRODUCTS_GUIDE.md` - Active guide
- `BENCHMARKS_QUICKSTART.md` - Current quickstart
- `BENCHMARK_EXPLANATIONS.md` - Current explanations

## Test Results Directories to Remove

- `benchmark_tmp/` - Temporary benchmark data
- `benchmarks_test/` - Old test results
- `final_test/` - Old final test
- `final_with_plots/` - Old results
- `results_proof/` - Old proof results
- `results_proof_50/` - Old proof results
- `test_batch_fix/` - Test data
- `test_fix/` - Test data
- `test_fix_large/` - Test data
- `test_isolated/` - Test data
- `test_isolated2/` - Test data
- `test_parallel_mmap_full/` - Test data

## Recommendation

**Phase 1: Remove clearly outdated files (safe)**
- All FINAL_* and *_SUMMARY.md files (historical snapshots)
- All *_FIX_*.md files (completed fixes)
- All test result directories

**Phase 2: Archive planning docs**
- Move B1 planning variants to docs/archive/
- Keep only canonical B1_GOAL.md, B1_GUIDE.md, B1_LONGTERM.md, B1_SHORTTERM.md

**Phase 3: Keep active documentation**
- ONDISK_* documents (current architecture)
- BENCHMARKS_QUICKSTART.md and BENCHMARK_EXPLANATIONS.md
- Main README and contribution guides
