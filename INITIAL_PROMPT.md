## Context & Mission

You are working on the **TopoBench Challenge 2025** submissions for:
- **B1 (Core)**: Large-Scale Inductive On-Disk Dataset Loader
- **B1 Bonus**: Large-Scale Transductive On-Disk Loader with Global Topology Preservation

The goal is to create a **production-grade, elegant extension** of the TopoBench framework that enables working with huge datasets (larger than available RAM but fitting on disk) in both inductive and transductive settings.

## Current State

We have implemented:
1. **Synthetic dataset classes** (in `topobench/data/datasets/`):
   - `synthetic_large_inductive_dataset.py` - Generates many graphs for inductive learning
   - `synthetic_large_transductive_dataset.py` - Generates single large graph for transductive learning

2. **Loaders** (in `topobench/data/loaders/`):
   - `synthetic_large_inductive_loader.py`
   - `synthetic_large_transductive_loader.py`

3. **On-disk preprocessing** (in `topobench/data/preprocessor/`):
   - `ondisk_inductive.py` - `OnDiskInductiveDataset` class
   - `ondisk_transductive.py` - `OnDiskTransductiveDataset` class

4. **Four validation scripts** (root directory - these are not yet conforming with the requirements):
   - `test_1_1_inmemory_inductive_FAILS_PROPER.py` - Shows in-memory approach fails (OOM)
   - `test_1_2_ondisk_inductive_WORKS_PROPER.py` - Shows on-disk approach succeeds
   - `test_2_1_transductive_FAILS_PROPER_V2.py` - Shows in-memory fails for transductive
   - `test_2_2_transductive_WORKS_PROPER_V2.py` - Shows on-disk succeeds for transductive

5. **Tutorial notebooks** (in `tutorials/`):
   - `tutorial_ondisk_inductive.ipynb`
   - `tutorial_ondisk_transductive.ipynb`

None of these files (and even the architecture) is set in stone and shuold be critically evaluated.

## Critical Architecture Questions

1. **Dataset Inheritance**: Our synthetic datasets currently inherit from `InMemoryDataset`, but should they inherit from `OnDiskDataset` (from `torch_geometric.data.on_disk_dataset.OnDiskDataset`) instead?

2. **Loader Architecture**: Should we create a base on-disk loader class (extending `AbstractLoader` from `topobench/data/loaders/base.py`) that all on-disk datasets can use?

3. **Preprocessor Integration**: How should `OnDiskInductiveDataset` and `OnDiskTransductiveDataset` fit into TopoBench's architecture? Should they be:
   - New preprocessor classes (like `PreProcessor` in `topobench/data/preprocessor/preprocessor.py`)?
   - Separate dataset classes?
   - Transforms?

4. **Workflow Alignment**: Users should be able to use the on-disk approach in the same/similar way as shown in `tutorial_model.ipynb` (using `PreProcessor`, then `TBDataloader`, then training).

## Your Task - Phase 1: Comprehensive Analysis

**Create 5 documents** (and ONLY these 5 + code and pytests):

### 1. **GOAL.md**
Comprehensive analysis containing:
- **What exists**: List all current implementations with brief descriptions
- **What works**: What has been tested and validated
- **What's missing**: Gaps, architectural issues, integration problems
- **What needs to be done**: Concrete action items with priorities
- Architecture recommendations (OnDiskDataset vs InMemoryDataset, loader design, preprocessor integration)

### 2. **LONGTERM.md**
Strategic document containing:
- **Vision**: Clear definition of the end goal for B1 & B1 Bonus
- **Architecture**: How the on-disk approach should integrate with TopoBench
- **Design principles**: Software engineering best practices we're following
- **Lessons learned**: Insights from current implementation
- **Progress tracking**: Broad milestones (not detailed tasks)
- Keep this concise but valuable - update it only when there are significant architectural insights

### 3. **SHORTTERM.md**
Tactical document containing:
- **Last 2 tasks completed**: What was just done (with brief outcomes)
- **Next task**: What needs to be done next (single, specific task)
- **Blockers/Notes**: Any issues or important context
- This document should be updated EVERY time work is completed - always shows last 2 done + next 1 to do

### 4. **GUIDE.md**
User-facing documentation:
- How to use the on-disk approach for inductive datasets
- How to use the on-disk approach for transductive datasets
- How to configure memory/disk constraints
- How to add custom datasets
- Code examples and best practices
- This should be the main documentation for end users

### 5. **PR_COMMIT.md**
Development tracking document with **two sections**:

**Section B1 (Core - Inductive):**
- List of files to commit with commit messages
- Ever-updating PR description/comment explaining the B1 submission
- What's implemented, how to test it, performance characteristics

**Section B1 Bonus (Transductive):**
- List of files to commit with commit messages
- Ever-updating PR description/comment explaining the B1 Bonus submission
- What's implemented, how to test it, performance characteristics

## Additional Requirements

After the analysis phase, you will need to implement:

1. **Unit tests** - Properly organized in the repository structure
2. **4 Training scripts** (refactor/optimize existing ones):
   - Configurable synthetic dataset size (user provides constraint, script calculates safe graph size accounting for higher-order structures)
   - Train with SCCNNCustom for 2 epochs
   - 1.1: Inductive in-memory FAILS (OOM)
   - 1.2: Inductive on-disk WORKS
   - 2.1: Transductive in-memory FAILS (OOM)
   - 2.2: Transductive on-disk WORKS
   - Be in TopoBench style (!)

3. **OGBN-products dataset** (transductive):
   - Make available with on-disk approach
   - Embed smoothly into current way TopoBench loads, processses (and ultimately trains) data
   - Create training script using TopoBench (on disk) workflow

4. **Real-world inductive dataset**:
   - Research and select appropriate dataset (search web, check OGBN, etc.)
   - Implement with on-disk approach
   - Embed smoothly into current way TopoBench loads, processses (and ultimately trains) data
   - Create training script

5. **Tutorial notebooks** (refine existing):
   - `tutorial_ondisk_inductive.ipynb` - Similar to existing tutorials but clear and concise
   - `tutorial_ondisk_transductive.ipynb` - Similar to existing tutorials but clear and concise

## Constraints

- **NO git operations** - Don't use git commands
- **Always use venv** - Use `.venv/bin/python` when executing Python commands
- **Clean code** - Follow PEP8, add type hints, write docstrings
- **Test everything** - Ensure code works before considering it done
- **Update documents** - Keep SHORTTERM.md and other docs updated as you work
- **No outdated info** - If you change direction, update all relevant documents

## Getting Started

1. **Read existing code** thoroughly:
   - Understand current dataset implementations
   - Understand current loader implementations  
   - Understand OnDiskInductiveDataset and OnDiskTransductiveDataset
   - Look at TopoBench's AbstractLoader, PreProcessor, InMemoryDataset patterns
   - Fully understand the TopoBench architecure, pipeline, workflow, design decisions etc
   - Study torch_geometric's OnDiskDataset

2. **Create GOAL.md** first with comprehensive analysis

3. **Create the other 4 documents** (LONGTERM.md, SHORTTERM.md, GUIDE.md, PR_COMMIT.md)

4. **Wait for next prompt** before implementing changes

## Success Criteria

Your submission will be evaluated on:
1. **Correctness** - Does it work? Is it mathematically sound? We want to preserve the full topology to stand out to other B1 submissions!
2. **Code Quality** - Readable, typed, documented, PEP8 compliant
3. **Documentation** - Clear docstrings, good README/guides
4. **Tests** - Robust unit tests with good coverage
5. **Integration** - Seamless integration with TopoBench framework
6. **Performance** - Constant memory usage for on-disk approach and suberb performance
7. **Deliverables** - Clear validation scripts showing what we are solving (something works that did not work before). This should be rigurous and undeniable proof. Do not settle for "we showed that all components work" or any other vague claim. Further jupyter tutorial notebooks showing how our solution smoothly embeds and elegantly extend the existing TopoBench framework.

Let's build something exceptional! Start by creating GOAL.md with your comprehensive analysis.
