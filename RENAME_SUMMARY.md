# Rename Summary: GeneratedInductiveDataset → OnDemandInductiveDataset

## Motivation

The name `GeneratedInductiveDataset` was misleading because it suggested the class was only for synthetic/procedurally generated data. However, it's actually for **any on-demand computation**, including:
- Synthetic data generation
- Subgraph extraction from massive graphs (Papers100M)
- Data augmentation on-the-fly
- Any computation cheaper than storing all samples

The new name `OnDemandInductiveDataset` accurately reflects its purpose: computing samples on-demand rather than loading from pre-saved files.

## Changes Made

### 1. Core Implementation
- **File**: `topobench/data/datasets/base_inductive.py`
  - Renamed class: `GeneratedInductiveDataset` → `OnDemandInductiveDataset`
  - Updated module docstring
  - Updated all class docstrings and examples

### 2. Exports
- **File**: `topobench/data/datasets/__init__.py`
  - Updated import statement
  - Updated `__all__` export list

### 3. Implementation Updates
- **File**: `topobench/data/datasets/ogbn_papers100m_ondisk.py`
  - Updated inheritance: `class OGBNPapers100MOnDiskDataset(OnDemandInductiveDataset)`

### 4. Tests
- **File**: `test/data/datasets/test_base_inductive.py`
  - Updated all test class names and references
  - All tests pass ✅

### 5. Documentation
- **File**: `CHOOSING_BASE_CLASSES.md`
  - Updated all references throughout
  - Updated decision tree
  - Updated examples

- **File**: `tutorials/tutorial_ondisk_inductive_final.ipynb`
  - Completely rewrote Cell 4 to be more precise and concise
  - Added explanation of `BaseOnDiskInductiveDataset`
  - Clarified benefits and use cases

## API Changes

### Before
```python
from topobench.data.datasets import GeneratedInductiveDataset

class MyDataset(GeneratedInductiveDataset):
    def _generate_sample(self, idx, rng):
        ...
```

### After
```python
from topobench.data.datasets import OnDemandInductiveDataset

class MyDataset(OnDemandInductiveDataset):
    def _generate_sample(self, idx, rng):
        ...
```

## Backward Compatibility

⚠️ **Breaking Change**: Code using `GeneratedInductiveDataset` will need to update imports.

**Migration**: Simple find-and-replace:
```bash
# In your code
sed -i 's/GeneratedInductiveDataset/OnDemandInductiveDataset/g' your_file.py
```

## Verification

All tests pass:
```bash
pytest test/data/datasets/test_base_inductive.py -v
# Result: 8 passed ✅
```

Import verification:
```python
from topobench.data.datasets import OnDemandInductiveDataset
# Works! ✅
```

## Summary

The rename improves API clarity:
- ✅ **FileBasedInductiveDataset**: Load from pre-saved files
- ✅ **OnDemandInductiveDataset**: Compute samples on-demand
- ✅ Names now accurately reflect their purposes
- ✅ Both inherit from `BaseOnDiskInductiveDataset` for all benefits

## Timeline

**Renamed**: November 24, 2025
**Reason**: Better clarity for Category B.1 submission and future users
**Status**: Complete and tested ✅
