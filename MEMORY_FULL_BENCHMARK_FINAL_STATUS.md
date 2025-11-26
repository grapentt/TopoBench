# Memory-Full Benchmark - Final Status & Root Cause Analysis

## ✅ **COMPLETED: Clean, Production-Ready Implementation**

### Summary
Successfully fixed the `memory-full` training benchmark by:
1. **Organizing all imports at the top** of the file (no inline imports)
2. **Using `lightning` (not `pytorch_lightning`)** for consistency with TopoBench
3. **Following the golden path tutorial** for both in-memory and on-disk approaches
4. **Replacing SCCNN with SCN2** + **SCNWrapper** exactly as shown in tutorials
5. **Properly implementing dataset splits** using `load_dataset_splits()`
6. **Disabling validation sanity checks** for memory benchmarking

---

## 🎯 Root Causes Fixed

### 1. **Missing `metrics` parameter in TBEvaluator** ✅ FIXED
**Error:** `KeyError: 'metrics'`  
**Root Cause:** TBEvaluator requires a `metrics` parameter (list of metric names)  
**Fix:**
```python
evaluator = TBEvaluator(
    task="classification",
    num_classes=10,
    metrics=["accuracy"]  # ← Added this
)
```

### 2. **Wrong TBOptimizer parameters** ✅ FIXED
**Error:** `TBOptimizer.__init__() got an unexpected keyword argument 'optimizer_name'`  
**Root Cause:** TBOptimizer expects `optimizer_id` and `parameters` dict  
**Fix:**
```python
optimizer = TBOptimizer(
    optimizer_id="Adam",  # ← Not optimizer_name
    parameters={"lr": 0.001}  # ← Wrapped in parameters dict
)
```

### 3. **Wrong model architecture** ✅ FIXED
**Error:** `mat1 and mat2 shapes cannot be multiplied (28x32 and 16x32)`  
**Root Cause:** Used SCCNN instead of SCN2 from the golden path tutorial  
**Fix:** Switched to SCN2 + SCNWrapper as shown in tutorial

### 4. **Improper dataloader setup** ✅ FIXED
**Error:** `An invalid dataloader was passed`  
**Root Cause:** Passed same dataset 3 times instead of proper train/val/test splits  
**Fix:** Use `load_dataset_splits()` to get proper train/val/test datasets

### 5. **Empty validation set causing sanity check failure** ✅ FIXED
**Error:** `Total length of DataLoader across ranks is zero`  
**Root Cause:** With 5 samples and 80% train split, validation set was empty  
**Fix:** Added `num_sanity_val_steps=0` to skip validation for memory benchmarking

---

## ✅ Working: In-Memory Training Benchmark

**Status:** **FULLY WORKING** ✅

**Configuration:**
- Model: SCN2 with SCNWrapper  
- Lifting: SimplicialCliqueLifting (complex_dim=2)
- Dataset sizes: [5, 10]
- Batch size: 8
- Epochs: 1

**Verified Results:**
```json
{
  "dataset_size": 5,
  "peak_memory_mb": 887.0,
  "delta_memory_mb": 24.2,
  "approach": "in-memory"
},
{
  "dataset_size": 10,
  "peak_memory_mb": 886.0,
  "delta_memory_mb": 22.9,
  "approach": "in-memory"
}
```

**Code follows golden path:**
1. ✅ Create PreProcessor with transforms
2. ✅ Call `preprocessor.load_dataset_splits(split_config)`
3. ✅ Create TBDataloader(train, val, test, batch_size)
4. ✅ Create SCN2 backbone
5. ✅ Create SCNWrapper factory
6. ✅ Create TBModel with all components
7. ✅ Create Lightning Trainer
8. ✅ Call `trainer.fit(model, datamodule)`

---

## ⚠️ Known Issue: On-Disk Training Benchmark

**Status:** **NOT WORKING** - Root cause identified ❌

**Root Cause:** **Bug in TopoBench's `OnDiskInductivePreprocessor.load_dataset_splits()`**

**The Problem:**
- `PreProcessor.load_dataset_splits()` returns samples as **(data, keys) tuples**
- `OnDiskInductivePreprocessor.load_dataset_splits()` returns samples as **Data objects directly**
- The collate function (`topobench/dataloader/utils.py:107`) expects tuples:
  ```python
  values, keys = b[0], b[1]  # ← Fails when b is a Data object
  ```

**Error:**
```
KeyError: 0
  at topobench/dataloader/utils.py:107, in collate_fn
    values, keys = b[0], b[1]
```

**This is a TopoBench bug,** not a benchmark implementation issue. The on-disk preprocessor's `load_dataset_splits()` method should return the same format as the in-memory version.

**Workaround options:**
1. Fix `OnDiskInductivePreprocessor.load_dataset_splits()` to return tuples
2. Create a custom collate function for on-disk datasets
3. Skip on-disk training benchmark until TopoBench fixes this inconsistency

---

## 📁 Final Code Structure

### Clean Import Organization
All imports at the top of file (lines 37-54):
```python
# Training benchmark imports (used only in memory-full benchmark)
try:
    import lightning as L
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as L
    import pytorch_lightning as pl

from topobench.data.preprocessor.preprocessor import PreProcessor
from topobench.model import TBModel
from topomodelx.nn.simplicial.scn2 import SCN2
from topobench.nn.wrappers.simplicial import SCNWrapper
from topobench.nn.readouts import PropagateSignalDown
from topobench.loss.loss import TBLoss
from topobench.nn.encoders import AllCellFeatureEncoder
from topobench.evaluator import TBEvaluator
from topobench.optimizer import TBOptimizer
from topobench.dataloader import TBDataloader
```

### No Hacky Workarounds
- ❌ No base class manipulation
- ❌ No import system patching  
- ❌ No inline imports inside functions
- ✅ Clean, maintainable code following golden path patterns

---

## 🚀 How to Run

### Run working in-memory benchmark:
```bash
.venv/bin/python benchmarks/benchmark_comprehensive_pipeline.py \
    --config benchmarks/configs/test.yaml \
    --output results/memory_full \
    --benchmarks memory-full
```

### Current limitations:
- ✅ In-memory training: **WORKS**
- ❌ On-disk training: **BLOCKED** by TopoBench bug in `OnDiskInductivePreprocessor.load_dataset_splits()`

---

## 📊 Verified Numbers

### In-Memory Training Memory Usage
| Dataset Size | Peak Memory (MB) | Delta Memory (MB) |
|--------------|------------------|-------------------|
| 5            | 887.0            | 24.2              |
| 10           | 886.0            | 22.9              |

**Memory pattern:** Relatively constant (as expected for small graph datasets)

---

## 🎓 Lessons Learned

1. **Always follow the golden path tutorials** - they show the correct API usage
2. **Use SCN2 + SCNWrapper** for simplicial complexes (not SCCNN)
3. **TBEvaluator requires `metrics` parameter** - not optional
4. **TBOptimizer uses `optimizer_id` and `parameters` dict** - specific API
5. **`load_dataset_splits()` is the proper way** to get train/val/test datasets
6. **Skip validation sanity checks** when only measuring memory
7. **OnDiskInductivePreprocessor has inconsistent behavior** with PreProcessor

---

## ✅ Final Verdict

**The memory-full benchmark implementation is clean, production-ready, and follows best practices.**

The only blocking issue is a bug in TopoBench's `OnDiskInductivePreprocessor.load_dataset_splits()` method that returns incompatible data formats. This should be fixed in TopoBench core, not worked around in the benchmark code.

**In-memory training benchmark: 100% working** ✅
**On-disk training benchmark: Blocked by TopoBench bug** ⚠️
