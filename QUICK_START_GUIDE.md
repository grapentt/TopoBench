# 🚀 QUICK START GUIDE
**TopoBench Challenge 2025 - Implementation Reference**

---

## 📁 FILE STRUCTURE TO CREATE

### Mission 1: Inductive Loader
```
topobench/data/preprocessor/
├── ondisk_inductive.py          # Main implementation (NEW)
└── ondisk_utils.py              # Helper functions (NEW)

test/data/preprocessor/
├── test_ondisk_inductive.py     # Unit tests (NEW)
└── test_ondisk_integration.py   # Integration tests (NEW)

test/pipeline/
└── test_pipeline.py             # Add OnDisk test case (MODIFY)
```

### Mission 2: Transductive Loader
```
topobench/data/preprocessor/
├── ondisk_transductive.py       # Main transductive loader (NEW)
├── structure_index.py           # Index builder/query (NEW)
└── streaming_cliques.py         # Clique enumeration (NEW)

topobench/data/preprocessor/backends/
├── __init__.py                  # Backend interface (NEW)
└── sqlite_backend.py            # SQLite3 implementation (NEW)

test/data/preprocessor/
├── test_ondisk_transductive.py  # Unit tests (NEW)
├── test_structure_index.py      # Index tests (NEW)
└── test_streaming_cliques.py    # Algorithm tests (NEW)
```

---

## 🔨 IMPLEMENTATION CHECKLIST

### Before Starting
- [ ] Activate virtual environment: `source venv/bin/activate`
- [ ] Review existing code: `topobench/data/preprocessor/preprocessor.py`
- [ ] Review loader base: `topobench/data/loaders/base.py`
- [ ] Review test examples: `test/data/`

### For Each New File
- [ ] Add module docstring
- [ ] Add all necessary imports
- [ ] Add type hints to all public methods
- [ ] Write NumPy-style docstrings
- [ ] Keep lines ≤ 79 characters
- [ ] Run: `pre-commit run --all-files` before committing

### For Each New Class
- [ ] Inherit from appropriate base class
- [ ] Implement all required abstract methods
- [ ] Add `__repr__` method
- [ ] Add usage example in docstring

### For Each New Method
- [ ] Type hints for parameters and return
- [ ] NumPy-style docstring
- [ ] Raise appropriate exceptions with helpful messages
- [ ] Add example in docstring if non-obvious

### For Each Test File
- [ ] Create corresponding test class
- [ ] Test happy path (success case)
- [ ] Test edge cases (empty, single item, max size)
- [ ] Test error cases (exceptions)
- [ ] Use fixtures for common setup
- [ ] Run: `pytest --cov=topobench test/ -v`

---

## 🎯 MILESTONE M1.1 - FIRST TASK

### Goal
Create `OnDiskInductiveDataset` that processes samples sequentially.

### Implementation Steps

**1. Create skeleton file:**
```bash
touch topobench/data/preprocessor/ondisk_inductive.py
```

**2. Basic structure:**
```python
"""On-disk dataset for inductive learning with large datasets."""

from pathlib import Path
from typing import Any, Optional

import torch
import torch_geometric
from torch.utils.data import Dataset
from tqdm import tqdm


class OnDiskInductiveDataset(Dataset):
    """Sequential disk-backed dataset for large inductive learning.
    
    This dataset processes samples one-by-one, saving each to disk
    immediately to avoid memory overflow during lifting operations.
    
    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Source dataset to process.
    data_dir : str
        Directory for storing processed samples.
    transforms_config : DictConfig, optional
        Transform configuration parameters.
    **kwargs : dict
        Additional arguments.
        
    Examples
    --------
    >>> from torch_geometric.datasets import TUDataset
    >>> source_data = TUDataset(root='/tmp', name='MUTAG')
    >>> dataset = OnDiskInductiveDataset(
    ...     dataset=source_data,
    ...     data_dir='/tmp/mutag_processed',
    ...     transforms_config=config
    ... )
    >>> sample = dataset[0]  # Loads from disk
    """
    
    def __init__(
        self,
        dataset: torch_geometric.data.Dataset,
        data_dir: str,
        transforms_config: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        # TODO: Implement initialization
        pass
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        # TODO: Implement
        pass
    
    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        """Load sample from disk.
        
        Parameters
        ----------
        idx : int
            Sample index.
            
        Returns
        -------
        torch_geometric.data.Data
            Loaded data sample.
        """
        # TODO: Implement
        pass
    
    def _process_samples(self) -> None:
        """Process all samples sequentially and save to disk."""
        # TODO: Implement sequential processing
        pass
```

**3. Create test file:**
```bash
touch test/data/preprocessor/test_ondisk_inductive.py
```

**4. Basic test structure:**
```python
"""Tests for OnDiskInductiveDataset."""

import pytest
import torch_geometric
from torch_geometric.datasets import TUDataset

from topobench.data.preprocessor.ondisk_inductive import (
    OnDiskInductiveDataset
)


class TestOnDiskInductiveDataset:
    """Test suite for OnDiskInductiveDataset."""
    
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create small test dataset."""
        # TODO: Create or load small dataset (e.g., 5 MUTAG graphs)
        pass
    
    def test_initialization(self, sample_dataset, tmp_path):
        """Test dataset initialization."""
        # TODO: Test dataset can be created
        pass
    
    def test_len(self, sample_dataset, tmp_path):
        """Test __len__ method."""
        # TODO: Test length is correct
        pass
    
    def test_getitem(self, sample_dataset, tmp_path):
        """Test __getitem__ method."""
        # TODO: Test sample can be loaded
        pass
    
    def test_sequential_processing(self, sample_dataset, tmp_path):
        """Test samples are processed one at a time."""
        # TODO: Test memory usage stays constant
        pass
```

**5. Run tests:**
```bash
pytest test/data/preprocessor/test_ondisk_inductive.py -v
```

---

## 🧪 TESTING COMMANDS

### Run Specific Test File
```bash
pytest test/data/preprocessor/test_ondisk_inductive.py -v
```

### Run Tests with Coverage
```bash
pytest --cov=topobench.data.preprocessor test/data/preprocessor/ -v
```

### Run All Tests
```bash
pytest test/ -v
```

### Check Coverage Report
```bash
pytest --cov=topobench --cov-report=html test/
# Open htmlcov/index.html in browser
```

### Run Pre-commit Hooks
```bash
pre-commit run --all-files
```

---

## 📝 CODE STYLE EXAMPLES

### Type Hints
```python
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

def process_sample(
    self,
    data: torch_geometric.data.Data,
    idx: int,
    save_path: Path
) -> None:
    """Process single sample and save to disk."""
    pass
```

### Docstring Template
```python
def method_name(self, param1: int, param2: str) -> List[str]:
    """One-line summary of what this does.

    Longer explanation if needed. Explain the why, not just
    the what. Reference papers or algorithms if relevant.

    Parameters
    ----------
    param1 : int
        Description of first parameter.
    param2 : str
        Description of second parameter.

    Returns
    -------
    List[str]
        Description of return value.

    Raises
    ------
    ValueError
        When invalid input is provided.
    FileNotFoundError
        When file doesn't exist.

    Examples
    --------
    >>> obj = MyClass()
    >>> result = obj.method_name(42, "hello")
    >>> print(result)
    ['processed', 'hello']

    Notes
    -----
    Any important implementation details or caveats.
    """
    pass
```

### Exception Handling
```python
if idx < 0 or idx >= len(self):
    raise IndexError(
        f"Index {idx} out of range for dataset of size {len(self)}"
    )

if not processed_path.exists():
    raise FileNotFoundError(
        f"Processed data not found at {processed_path}. "
        f"Run processing first by creating dataset with "
        f"transforms_config parameter."
    )
```

---

## 🔍 DEBUGGING TIPS

### Memory Profiling
```python
# Install memory_profiler
pip install memory_profiler

# Add decorator to function
from memory_profiler import profile

@profile
def _process_samples(self):
    # Your code here
    pass

# Run with profiling
python -m memory_profiler your_script.py
```

### Logging
```python
import logging

logger = logging.getLogger(__name__)

def process(self):
    logger.info(f"Processing {len(self.dataset)} samples")
    for idx in range(len(self.dataset)):
        logger.debug(f"Processing sample {idx}")
        # Process...
```

### Interactive Testing
```python
# In IPython or Jupyter
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductiveDataset
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp', name='MUTAG')
ondisk = OnDiskInductiveDataset(dataset, '/tmp/test')

# Check what's happening
print(ondisk)
print(len(ondisk))
sample = ondisk[0]
print(sample)
```

---

## 🎓 COMMON PATTERNS FROM EXISTING CODE

### Pattern 1: Transform Hashing
```python
# See: topobench/data/preprocessor/preprocessor.py
from topobench.data.utils import make_hash

params_hash = make_hash(transforms_parameters)
processed_dir = os.path.join(data_dir, repo_name, f"{params_hash}")
```

### Pattern 2: Progress Bars
```python
from tqdm import tqdm

for idx in tqdm(range(len(dataset)), desc="Processing samples"):
    # Process sample
    pass
```

### Pattern 3: Safe File I/O
```python
from pathlib import Path

save_path = Path(processed_dir) / f"sample_{idx:06d}.pt"
save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, save_path)
```

### Pattern 4: Dataset Iteration
```python
# Iterate through PyG dataset safely
for idx in range(len(dataset)):
    data = dataset[idx]
    # Process data
```

---

## 📊 VERIFICATION CHECKLIST

### Before Committing
- [ ] All tests pass: `pytest test/ -v`
- [ ] Coverage ≥93%: `pytest --cov=topobench test/`
- [ ] Pre-commit passes: `pre-commit run --all-files`
- [ ] No TODO comments in committed code
- [ ] All docstrings complete
- [ ] Examples in docstrings tested

### Before Submitting PR
- [ ] MASTER_PLAN.md updated with progress
- [ ] All milestones in phase completed
- [ ] Pipeline test added and passing
- [ ] Code reviewed (self-review)
- [ ] No debug print statements
- [ ] Error messages are user-friendly

---

## 🆘 TROUBLESHOOTING

### Import Errors
```bash
# Ensure you're in the right environment
source venv/bin/activate

# Ensure TopoBench is installed in editable mode
pip install -e .
```

### Test Discovery Issues
```bash
# Make sure __init__.py exists in test directories
touch test/data/preprocessor/__init__.py
```

### Hydra State Issues
```python
# In test setup
def setup_method(self):
    """Clear Hydra state before each test."""
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
```

---

## 📚 REFERENCE FILES TO READ

### Before Implementing Inductive Loader
1. `topobench/data/preprocessor/preprocessor.py` - Current implementation
2. `topobench/data/loaders/base.py` - Loader interface
3. `topobench/data/loaders/graph/tu_datasets.py` - Simple loader example
4. `test/data/load/test_datasetloaders.py` - Test examples

### Before Implementing Transductive Loader
1. `topobench/data/utils.py` - Utility functions
2. Review NetworkX clique finding documentation
3. SQLite3 Python documentation

---

## 🎯 SUCCESS METRICS

### Milestone M1.1 Complete When:
- [ ] `OnDiskInductiveDataset` class exists
- [ ] Sequential processing implemented
- [ ] Can save and load samples from disk
- [ ] Memory usage verified constant
- [ ] Unit tests pass with ≥90% coverage
- [ ] No ruff/flake8 warnings

### Ready to Move to M1.2 When:
- [ ] M1.1 complete
- [ ] Code reviewed
- [ ] All TODOs resolved
- [ ] Examples in docstrings work

---

**Status:** Ready for Implementation  
**First Task:** Create `ondisk_inductive.py` skeleton  
**Time Estimate:** 2-3 hours for M1.1  
**Next Review:** After M1.1 completion
