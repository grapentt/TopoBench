# On-Disk Training Bug Analysis

## Executive Summary

**Status:** Root cause identified ✓  
**Impact:** On-disk inductive training fails with `TypeError` when accessing split datasets  
**Solution:** Create `LazyDataloadDataset` wrapper for memory-efficient split compatibility

---

## Root Cause Analysis

### The Bug

When using `OnDiskInductivePreprocessor.load_dataset_splits()` for training:

```python
dataset_train, dataset_val, dataset_test = ondisk_dataset.load_dataset_splits(split_config)
datamodule = TBDataloader(dataset_train, dataset_val, dataset_test, batch_size=8)
trainer.fit(model, datamodule)  # ← FAILS HERE
```

**Error:**
```python
File "topobench/dataloader/utils.py", line 107, in collate_fn
    values, keys = b[0], b[1]  # ← Expects tuple, gets Data object
TypeError: 'Data' object is not subscriptable
```

### Data Flow Analysis

#### Working Path (PreProcessor - In-Memory)

1. **PreProcessor.load_dataset_splits()** → calls `load_inductive_splits(self, split_params)`
2. **load_inductive_splits()** → returns `DataloadDataset` objects (when `use_lazy=False`)
3. **DataloadDataset.__getitem__(idx)** → calls `self.get(idx)`
4. **DataloadDataset.get(idx)** → returns `([data[key] for key in keys], keys)` ✓
5. **collate_fn()** → receives `(values, keys)` tuple ✓

#### Broken Path (OnDiskInductivePreprocessor)

1. **OnDiskInductivePreprocessor.load_dataset_splits()** → calls `load_inductive_splits(self, split_params, use_lazy=True)`
2. **load_inductive_splits()** → returns `LazySubset` objects (when `use_lazy=True`)
3. **LazySubset.__getitem__(idx)** → returns `self.dataset[actual_idx]` directly
4. **OnDiskInductivePreprocessor.__getitem__(idx)** → returns `Data` object ✗
5. **collate_fn()** → receives `Data` object instead of tuple ✗

### Architecture Components

**Key Files:**
- `topobench/data/preprocessor/ondisk_inductive.py` - OnDiskInductivePreprocessor (line 1064)
- `topobench/data/datasets/_lazy.py` - LazySubset (line 101)
- `topobench/dataloader/dataload_dataset.py` - DataloadDataset (line 36-37)
- `topobench/dataloader/utils.py` - collate_fn (line 107)
- `topobench/data/utils/split_utils.py` - load_inductive_splits (line 358-362)

### Why This Design Mismatch Exists

**PreProcessor (In-Memory):**
- Loads all data into memory
- Uses `DataloadDataset` wrapper
- DataloadDataset.get() unpacks Data objects into `(values, keys)` tuples
- Memory usage: O(n) - all samples in RAM

**OnDiskInductivePreprocessor:**
- Stores data on disk for O(1) memory
- Uses `LazySubset` for O(1) memory splits
- LazySubset just forwards to dataset's `__getitem__`
- Returns Data objects directly (no unpacking)
- Memory usage: O(1) - samples loaded on-demand

**The Problem:** LazySubset doesn't unpack Data objects into tuples like DataloadDataset does.

---

## Solution Design

### Requirements

1. **Memory Efficiency:** Maintain O(1) memory usage (no loading all samples)
2. **Compatibility:** Work with existing `collate_fn` expecting tuples
3. **Performance:** Minimal overhead, no unnecessary data copies
4. **Integration:** Seamless drop-in replacement in existing code
5. **Backward Compatibility:** Don't break existing OnDiskInductivePreprocessor usage

### Proposed Solution: LazyDataloadDataset

Create a new wrapper class that combines:
- LazySubset's O(1) memory efficiency (stores indices only)
- DataloadDataset's tuple-returning interface

**Implementation Strategy:**

```python
class LazyDataloadDataset(torch_geometric.data.Dataset):
    """Memory-efficient dataset split compatible with DataloadDataset interface.
    
    Combines LazySubset's O(1) memory with DataloadDataset's tuple interface.
    Stores only indices, loads samples on-demand, returns (values, keys) tuples.
    """
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices) if not isinstance(indices, list) else indices
    
    def len(self):
        return len(self.indices)
    
    def get(self, idx):
        # Load sample via LazySubset pattern
        actual_idx = self.indices[idx]
        data = self.dataset[actual_idx]
        
        # Unpack like DataloadDataset
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)
```

**Why This Works:**
- PyG's Dataset.__getitem__ calls self.get() automatically
- Memory: O(1) - only stores indices, loads samples on-demand
- Compatible: Returns tuples like DataloadDataset
- Performance: Single sample load per access (no extra copies)

### Implementation Changes

**File:** `topobench/data/datasets/_lazy.py`

Add new class `LazyDataloadDataset` alongside `LazySubset`.

**File:** `topobench/data/utils/split_utils.py`

Modify `assign_train_val_test_mask_to_graphs()` to use `LazyDataloadDataset` when `use_lazy=True`:

```python
if use_lazy:
    from topobench.data.datasets import LazyDataloadDataset
    return (
        LazyDataloadDataset(dataset, split_idx["train"]),
        LazyDataloadDataset(dataset, split_idx["valid"]),
        LazyDataloadDataset(dataset, split_idx["test"]),
    )
```

**File:** `topobench/data/datasets/__init__.py`

Export new class for easy importing.

### Alternative Solutions Considered

**Option 1:** Wrap LazySubset in DataloadDataset
- ❌ DataloadDataset.__init__ expects list of Data objects (defeats O(1) memory)

**Option 2:** Modify LazySubset.__getitem__ to return tuples
- ❌ Breaks backward compatibility for direct LazySubset usage

**Option 3:** Modify OnDiskInductivePreprocessor.__getitem__ to return tuples
- ❌ Breaks backward compatibility for anyone using the dataset directly

**Option 4:** Create special collate_fn variant for on-disk datasets
- ❌ Increases code complexity, requires users to know which collate to use

**Option 5 (Chosen):** Create LazyDataloadDataset wrapper
- ✅ Clean separation of concerns
- ✅ Maintains O(1) memory
- ✅ Backward compatible
- ✅ Works with existing collate_fn

---

## Performance Impact

**Memory:**
- Before: O(1) but broken ✗
- After: O(1) and working ✓
- Overhead: Negligible (~few bytes per dataset for index list)

**Speed:**
- No additional data copies
- Single sample load per access (same as before)
- Tuple unpacking overhead: < 1 microsecond per sample (negligible)

**Training:**
- Enable on-disk training for large datasets
- Prevents OOM errors
- Maintains training speed parity with in-memory approach

---

## Testing Strategy

**Unit Tests:**
1. Test LazyDataloadDataset returns correct tuple format
2. Test memory usage stays O(1)
3. Test integration with collate_fn
4. Test split functionality with OnDiskInductivePreprocessor

**Integration Tests:**
1. End-to-end training with on-disk splits
2. Verify memory efficiency during training
3. Compare accuracy with in-memory baseline

**Regression Tests:**
1. Ensure existing LazySubset usage still works
2. Verify DataloadDataset behavior unchanged
3. Check PreProcessor splits still work

---

## Implementation Checklist

- [ ] Create `LazyDataloadDataset` class in `_lazy.py`
- [ ] Update `assign_train_val_test_mask_to_graphs()` to use new class
- [ ] Export new class in `__init__.py`
- [ ] Add unit tests for LazyDataloadDataset
- [ ] Add integration test for on-disk training
- [ ] Update documentation and docstrings
- [ ] Run full test suite
- [ ] Verify benchmark_comprehensive_pipeline.py works
- [ ] Update tutorials if needed

---

## Future Considerations

**Potential Optimizations:**
1. Add caching to LazyDataloadDataset (like OnDiskInductivePreprocessor has)
2. Support batch-aware prefetching for training speedup
3. Add memory profiling utilities

**Architecture Improvements:**
1. Consider unifying DataloadDataset and LazyDataloadDataset interfaces
2. Explore zero-copy optimizations for tuple creation
3. Add type hints throughout data loading pipeline

---

## References

**Related Files:**
- `topobench/data/preprocessor/ondisk_inductive.py:1019-1064`
- `topobench/data/datasets/_lazy.py:74-101`
- `topobench/dataloader/dataload_dataset.py:22-37`
- `topobench/dataloader/utils.py:85-171`
- `topobench/data/utils/split_utils.py:183-239, 302-364`

**Related Issues:**
- MEMORY_FULL_BENCHMARK_FINAL_STATUS.md (lines 97-109)
- Category B.1 Challenge: Large-Scale Inductive Data Infrastructure
