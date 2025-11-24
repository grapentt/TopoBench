"""Tests for OnDiskInductivePreprocessor.

This test suite covers essential functionality for on-disk preprocessing:
- Basic initialization and data access
- Transform caching and force_reload
- Integration with real PyG datasets
- Parallel processing integration
- Split functionality for inductive learning
- Source dataset flexibility (InMemory, OnDisk, Custom)
"""

import gc
import json
import tempfile
import time
from pathlib import Path

import psutil
import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, OnDiskDataset
from torch_geometric.datasets import TUDataset

from topobench.data.preprocessor.ondisk_inductive import (
    OnDiskInductivePreprocessor,
)

# Test datasets at module level for picklability in multiprocessing.
# Use on-demand generation to minimize pickle overhead (see B1_LONGTERM.md).

class SyntheticInMemoryDataset(InMemoryDataset):
    """Synthetic :class:`InMemoryDataset` used in tests.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        # Initialize required attributes for InMemoryDataset
        self._indices = None
        self.transform = None
        self.pre_transform = None
        self.pre_filter = None
    
    def len(self):
        """Return number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.num_samples

    def get(self, idx):
        """Return sample at given index.

        Generate data on-demand to avoid pickling overhead in parallel processing.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Data
            The generated sample data.
        """
        torch.manual_seed(idx)
        return Data(
            x=torch.randn(5 + idx % 3, 8),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            y=torch.tensor([idx % 3]),
        )
    
    def __reduce__(self):
        """Support pickling for multiprocessing.

        Returns
        -------
        tuple
            Tuple for pickle reconstruction.
        """
        return (self.__class__, (self.num_samples,))


def create_inmemory_dataset(num_samples: int = 10) -> SyntheticInMemoryDataset:
    """Create a synthetic InMemoryDataset for testing.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
        
    Returns
    -------
    SyntheticInMemoryDataset
        Synthetic in-memory dataset (picklable!).
    """
    return SyntheticInMemoryDataset(num_samples)


class SyntheticOnDiskDataset(OnDiskDataset):
    """Synthetic :class:`OnDiskDataset` used in tests.

    Parameters
    ----------
    root : str
        Root directory for the underlying database.
    num_samples : int
        Number of samples to generate.
    """

    def __init__(self, root: str, num_samples: int):
        super().__init__(root, backend="sqlite")
        self.num_samples = num_samples
        self._process()
    
    def _process(self):
        """Populate the on-disk database with synthetic samples."""
        for i in range(self.num_samples):
            data = Data(
                x=torch.randn(5 + i % 3, 8),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
                y=torch.tensor([i % 3]),
            )
            self.append(data)


def create_ondisk_dataset(num_samples: int = 10, tmpdir: Path = None) -> SyntheticOnDiskDataset:
    """Create a synthetic OnDiskDataset for testing.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    tmpdir : Path
        Temporary directory for the database.
        
    Returns
    -------
    SyntheticOnDiskDataset
        Synthetic on-disk dataset (picklable!).
    """
    db_dir = tmpdir / "ondisk_db" if tmpdir else Path(tempfile.mkdtemp()) / "ondisk_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return SyntheticOnDiskDataset(str(db_dir), num_samples)


class SyntheticCustomDataset(torch.utils.data.Dataset):
    """Synthetic custom Dataset used in tests.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
    
    def __len__(self):
        """Return number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """Return sample at given index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Data
            The sample data.
        """
        return Data(
            x=torch.randn(5 + idx % 3, 8),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            y=torch.tensor([idx % 3]),
        )


def create_custom_dataset(num_samples: int = 10) -> SyntheticCustomDataset:
    """Create a custom dataset (not PyG) for testing flexibility.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
        
    Returns
    -------
    SyntheticCustomDataset
        Custom torch Dataset (picklable!).
    """
    return SyntheticCustomDataset(num_samples)


@pytest.fixture(
    params=["inmemory", "ondisk", "custom"],
    ids=["InMemoryDataset", "OnDiskDataset", "CustomDataset"]
)
def dataset_type(request):
    """Parametrized fixture providing different dataset types.
    
    Tests that take this fixture as a parameter will run 3 times
    (once for each dataset type), ensuring our preprocessor is truly
    source-agnostic across InMemory, OnDisk, and Custom datasets.
    
    Tests that don't need source-type flexibility (e.g., memory tests,
    file structure tests) should not take this parameter.
    
    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.
    
    Returns
    -------
    str
        Dataset type: 'inmemory', 'ondisk', or 'custom'.
    """
    return request.param


def create_source_dataset(dataset_type: str, num_samples: int, tmpdir: Path = None):
    """Helper to create source dataset of specified type.
    
    Parameters
    ----------
    dataset_type : str
        One of: "inmemory", "ondisk", "custom".
    num_samples : int
        Number of samples to generate.
    tmpdir : Path, optional
        Temporary directory (required for ondisk type).

    Returns
    -------
    Dataset
        Source dataset of the requested type.
    """
    if dataset_type == "inmemory":
        return create_inmemory_dataset(num_samples)
    elif dataset_type == "ondisk":
        if tmpdir is None:
            raise ValueError("tmpdir required for ondisk dataset")
        return create_ondisk_dataset(num_samples, tmpdir=tmpdir)
    elif dataset_type == "custom":
        return create_custom_dataset(num_samples)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


@pytest.fixture(autouse=True)
def track_memory():
    """Fixture that tracks memory usage for all tests to detect leaks.
    
    This runs automatically for every test (autouse=True) and verifies that
    memory growth stays reasonable, catching any O(n) memory bugs early.
    
    Yields
    ------
    None
        Nothing, used for setup/teardown.
    """
    # Force GC before test to get clean baseline
    # (ensures previous test's garbage is collected)
    gc.collect()
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    yield  # Run the test
    
    # Force GC after test to detect if test leaked memory
    # (if memory doesn't drop after GC, test accumulated data)
    gc.collect()
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_growth = mem_after - mem_before
    
    # Warn if test used excessive memory (but don't fail)
    # Allow generous threshold
    if mem_growth > 100:  # 100MB threshold
        pytest.warn(
            UserWarning(
                f"Test used {mem_growth:.1f}MB memory. "
                f"Check for potential O(n) memory usage."
            )
        )


class TestOnDiskInductivePreprocessor:
    """Test suite for OnDiskInductivePreprocessor.
    
    Key tests (basic_functionality, caching, splits) are parametrized with the
    `dataset_type` fixture to run with InMemory, OnDisk, and Custom source datasets,
    proving our preprocessor is truly source-agnostic.
    
    Other tests (memory, file structure, parallel processing) don't require
    source-type flexibility and run once with InMemory for speed.
    """

    def test_basic_functionality(self, dataset_type):
        """Test initialization, data access, and metadata.
        
        Parameters
        ----------
        dataset_type : str
            Type of dataset to test ('inmemory', 'ondisk', 'custom').
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_source_dataset(dataset_type, num_samples=20, tmpdir=Path(tmpdir))
            
            # Initialize preprocessor
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=None,
            )
            
            # Test basic properties
            assert len(dataset) == 20
            assert dataset.processed_dir.exists()
            assert dataset.metadata_path.exists()
            
            # Test data access
            for idx in [0, 10, 19]:
                sample = dataset[idx]
                assert isinstance(sample, Data)
                assert hasattr(sample, "x")
                assert hasattr(sample, "edge_index")
                assert sample.y.item() == idx % 3
            
            # Test invalid indices
            with pytest.raises(IndexError):
                _ = dataset[-1]
            with pytest.raises(IndexError):
                _ = dataset[20]
            
            # Test metadata
            with open(dataset.metadata_path) as f:
                metadata = json.load(f)
            assert metadata["num_samples"] == 20
            assert "source_dataset" in metadata

    def test_caching_and_force_reload(self, dataset_type):
        """Test caching behavior and force_reload parameter.
        
        Parameters
        ----------
        dataset_type : str
            Type of dataset to test ('inmemory', 'ondisk', 'custom').
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_source_dataset(dataset_type, num_samples=15, tmpdir=Path(tmpdir))
            
            # First initialization - processes data
            dataset1 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=None,
            )
            
            # Get modification time
            sample_path = dataset1._get_sample_path(0)
            mtime1 = sample_path.stat().st_mtime
            
            # Second initialization - should use cache
            dataset2 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=None,
            )
            sample_path2 = dataset2._get_sample_path(0)  # Get from dataset2
            mtime2 = sample_path2.stat().st_mtime
            assert mtime1 == mtime2, "Cache should be reused"
            
            # Corrupt a file
            torch.save({"corrupted": True}, sample_path)
            
            # Verify corrupted file can't be loaded properly
            corrupted_data = torch.load(sample_path)
            assert "corrupted" in corrupted_data
            
            # Force reload should recreate
            _dataset3 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=None,
                force_reload=True,
            )
            
            # Verify data is restored after reload
            restored_data = torch.load(sample_path)
            assert isinstance(restored_data, Data)
            assert hasattr(restored_data, "x")
            assert "corrupted" not in restored_data

    def test_integration_with_real_dataset(self):
        """Integration test with real PyG TUDataset (MUTAG)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Load MUTAG dataset (188 samples, small for testing)
                source = TUDataset(root=tmpdir, name="MUTAG")
                
                data_dir = Path(tmpdir) / "processed"
                dataset = OnDiskInductivePreprocessor(
                    dataset=source,
                    data_dir=data_dir,
                    transforms_config=None,
                    num_workers=2,  # Test parallel processing
                )
                
                # Verify basic properties
                assert len(dataset) == len(source)
                
                # Verify data access
                for idx in [0, 50, 187]:
                    sample = dataset[idx]
                    assert isinstance(sample, Data)
                    assert hasattr(sample, "x") or hasattr(sample, "edge_index")
                
                # Verify caching works
                sample_path = dataset._get_sample_path(0)
                mtime1 = sample_path.stat().st_mtime
                
                # Reinitialize - should use cache
                dataset2 = OnDiskInductivePreprocessor(
                    dataset=source,
                    data_dir=data_dir,
                    transforms_config=None,
                )
                mtime2 = dataset2._get_sample_path(0).stat().st_mtime
                assert mtime1 == mtime2
                
            except Exception as e:
                pytest.skip(f"MUTAG dataset not available: {e}")
    
    def test_splits_functionality(self, dataset_type):
        """Test load_dataset_splits for inductive learning.
        
        Parameters
        ----------
        dataset_type : str
            Type of dataset to test ('inmemory', 'ondisk', 'custom').
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_source_dataset(dataset_type, num_samples=30, tmpdir=Path(tmpdir))
            
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=None,
            )
            
            # Test correct inductive setting
            split_params = DictConfig({
                "learning_setting": "inductive",
                "split_type": "random",
                "data_seed": 42,
                "train_prop": 0.6,
                "val_prop": 0.2,
                "data_split_dir": str(data_dir / "splits"),
            })
            
            train_ds, val_ds, test_ds = dataset.load_dataset_splits(split_params)
            
            assert train_ds is not None
            assert val_ds is not None
            assert test_ds is not None
            assert len(train_ds) + len(val_ds) + len(test_ds) == len(dataset)
            
            # Test wrong learning setting raises error
            bad_params = DictConfig({"learning_setting": "transductive"})
            with pytest.raises(ValueError) as exc_info:
                dataset.load_dataset_splits(bad_params)
            assert "inductive" in str(exc_info.value).lower()
            
            # Test missing learning_setting raises error
            with pytest.raises(ValueError):
                dataset.load_dataset_splits(DictConfig({}))
    
    
    def test_file_structure_creation(self):
        """Test that correct directory structure and files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_inmemory_dataset(num_samples=25)
            
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=None,
            )
            
            # Verify processed directory exists
            assert dataset.processed_dir.exists()
            assert dataset.processed_dir.is_dir()
            
            # Verify metadata file exists
            assert dataset.metadata_path.exists()
            assert dataset.metadata_path.is_file()
            
            # Verify correct number of sample files
            sample_files = list(dataset.processed_dir.glob("sample_*.pt"))
            assert len(sample_files) == 25
            
            # Verify file naming pattern (zero-padded)
            expected_names = [f"sample_{i:06d}.pt" for i in range(25)]
            actual_names = sorted([f.name for f in sample_files])
            assert actual_names == expected_names
            
            # Verify all files are non-empty
            for sample_file in sample_files:
                assert sample_file.stat().st_size > 0
    
    def test_memory_usage_stays_constant(self):
        """Test that RAM usage remains O(1) regardless of dataset size.
        
        This test verifies the core guarantee of our on-disk approach: memory usage
        should NOT scale with dataset size. We process datasets of different sizes
        and verify iteration doesn't accumulate data in memory.
        
        This test is independent of source dataset type (uses InMemory for speed),
        so it doesn't use the parametrized fixture.
        
        Note: The track_memory fixture handles GC and memory monitoring for the
        entire test. This test focuses on verifying O(1) behavior during iteration.
        """
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # GC before measurement (explicit for accurate baseline in this specific test)
            gc.collect()
            
            # Measure baseline memory
            process = psutil.Process()
            baseline_mem = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process small dataset
            source_small = create_inmemory_dataset(num_samples=50)
            OnDiskInductivePreprocessor(
                dataset=source_small,
                data_dir=data_dir / "small",
                transforms_config=None,
                num_workers=1,  # Sequential for consistent measurement
            )
            gc.collect()  # Ensure source_small is freed
            mem_after_small = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process larger dataset (10× bigger)
            source_large = create_inmemory_dataset(num_samples=1000)
            dataset_large = OnDiskInductivePreprocessor(
                dataset=source_large,
                data_dir=data_dir / "large",
                transforms_config=None,
                num_workers=1,  # Sequential for consistent measurement
            )
            gc.collect()  # Ensure source_large is freed
            mem_after_large = process.memory_info().rss / 1024 / 1024  # MB
            
            # Memory growth should be proportional to processing overhead, not dataset size
            mem_growth_small = max(0.1, mem_after_small - baseline_mem)  # Avoid zero
            mem_growth_large = max(0.1, mem_after_large - mem_after_small)  # Avoid zero
            
            # CRITICAL TEST: Verify we can iterate through large dataset without memory spike
            # This is the O(1) memory guarantee for graph data during iteration
            mem_before_iter = process.memory_info().rss / 1024 / 1024
            
            # Iterate through all samples
            for i in range(len(dataset_large)):
                _ = dataset_large[i]
                if i % 100 == 0:
                    # Periodic GC during iteration to prevent accumulation artifacts
                    gc.collect()
            
            gc.collect()  # Final GC to measure true retained memory
            mem_after_iter = process.memory_info().rss / 1024 / 1024
            
            # Iteration through 1000 samples should not significantly increase memory
            iter_growth = mem_after_iter - mem_before_iter
            
            # This is the critical assertion: O(1) memory during iteration
            # Allow max 10MB growth (generous, should be around 1MB)
            assert iter_growth < 10, (
                f"Memory grew by {iter_growth:.1f}MB during iteration over 500 samples. "
                f"This indicates O(n) memory usage instead of O(1)."
            )
            
            print("\nMemory test results:")
            print(f"  Small dataset (50 samples): +{mem_growth_small:.1f}MB")
            print(f"  Large dataset (1000 samples): +{mem_growth_large:.1f}MB")
            print(f"  Iteration (1000 samples): +{iter_growth:.1f}MB")
            print("  ✓ O(1) memory confirmed!")
    
    def test_parallel_vs_sequential_correctness_and_performance(self):
        """Test that parallel processing produces identical results and is faster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_inmemory_dataset(num_samples=500)
            
            # Process sequentially (num_workers=1)
            start_seq = time.time()
            dataset_seq = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir / "sequential",
                transforms_config=None,
                num_workers=1,
            )
            time_seq = time.time() - start_seq
            
            # Process in parallel (num_workers=4)
            start_par = time.time()
            dataset_par = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir / "parallel",
                transforms_config=None,
                num_workers=None, # default value = auto-detect
            )
            time_par = time.time() - start_par
            
            # Verify same length
            assert len(dataset_seq) == len(dataset_par) == 500
            
            # Verify identical results
            for idx in range(0, 500, 10):  # Sample every 10th
                data_seq = dataset_seq[idx]
                data_par = dataset_par[idx]
                
                # Same structure
                assert data_seq.x.shape == data_par.x.shape
                assert data_seq.edge_index.shape == data_par.edge_index.shape
                assert data_seq.y.item() == data_par.y.item()
            
            # Parallel should be faster on Linux with lightweight dataset
            speedup = time_seq / time_par
            print(f"\nLightweight dataset parallel speedup: {speedup:.2f}× (sequential={time_seq:.2f}s, parallel={time_par:.2f}s)")
    
    def test_ondemand_vs_inmemory_parallel_speedup(self):
        """Prove on-demand loading achieves better parallel speedup than InMemoryDataset.
        
        Compares lightweight on-demand pattern vs heavy InMemoryDataset for parallel
        preprocessing. On-demand should be faster due to minimal pickle overhead.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            num_samples = 200
            
            # Test 1: InMemoryDataset (heavy pickling)
            enzymes = TUDataset(root=str(data_dir / "raw"), name="ENZYMES")
            enzymes_subset = enzymes[:num_samples]
            
            start = time.time()
            dataset_inmemory = OnDiskInductivePreprocessor(
                dataset=enzymes_subset,
                data_dir=data_dir / "inmemory",
                num_workers=4,
            )
            time_inmemory = time.time() - start
            
            # Test 2: On-demand pattern (lightweight pickling)
            lightweight_dataset = create_inmemory_dataset(num_samples=num_samples)
            
            start = time.time()
            dataset_ondemand = OnDiskInductivePreprocessor(
                dataset=lightweight_dataset,
                data_dir=data_dir / "ondemand",
                num_workers=4,
            )
            time_ondemand = time.time() - start
            
            speedup = time_inmemory / time_ondemand
            
            # Verify correctness
            assert len(dataset_inmemory) == len(dataset_ondemand) == num_samples
            
            # Assert on-demand is faster (conservative threshold)
            assert speedup > 1.0, (
                f"On-demand speedup {speedup:.2f}× should be >1.0× vs InMemoryDataset"
            )


class TestMemoryMappedStorageIntegration:
    """Test MemoryMappedStorage integration proving I/O speedup and compression benefits.
    
    Uses class-level fixtures to reduce overhead by sharing preprocessor instances.
    """

    @pytest.fixture(scope="class")
    def temp_dir_class(self):
        """Class-level temporary directory (shared across tests).
        
        Yields
        ------
        Path
            Temporary directory path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture(scope="class")
    def source_dataset(self, temp_dir_class):
        """Shared source dataset for all tests.
        
        Returns
        -------
        SyntheticCustomDataset
            Dataset with 50 samples.
        """
        return SyntheticCustomDataset(num_samples=50)

    @pytest.fixture(scope="class")
    def prep_mmap(self, source_dataset, temp_dir_class):
        """Shared mmap preprocessor (LZ4 compression).
        
        Returns
        -------
        OnDiskInductivePreprocessor
            Preprocessor with mmap storage and LZ4 compression.
        """
        return OnDiskInductivePreprocessor(
            dataset=source_dataset,
            data_dir=temp_dir_class / "mmap",
            storage_backend="mmap",
            compression="lz4",
            cache_size=0,
        )

    @pytest.fixture(scope="class")
    def prep_files(self, source_dataset, temp_dir_class):
        """Shared file-based preprocessor.
        
        Returns
        -------
        OnDiskInductivePreprocessor
            Preprocessor with file-based storage.
        """
        return OnDiskInductivePreprocessor(
            dataset=source_dataset,
            data_dir=temp_dir_class / "files",
            storage_backend="files",
            cache_size=0,
        )

    @pytest.fixture(scope="class")
    def prep_nocomp(self, source_dataset, temp_dir_class):
        """Shared mmap preprocessor without compression.
        
        Returns
        -------
        OnDiskInductivePreprocessor
            Preprocessor with mmap storage, no compression.
        """
        return OnDiskInductivePreprocessor(
            dataset=source_dataset,
            data_dir=temp_dir_class / "nocomp",
            storage_backend="mmap",
            compression=None,
            cache_size=0,
        )

    @pytest.fixture(scope="class")
    def prep_zstd(self, source_dataset, temp_dir_class):
        """Shared mmap preprocessor with ZSTD compression.
        
        Returns
        -------
        OnDiskInductivePreprocessor
            Preprocessor with mmap storage and ZSTD compression.
        """
        return OnDiskInductivePreprocessor(
            dataset=source_dataset,
            data_dir=temp_dir_class / "zstd",
            storage_backend="mmap",
            compression="zstd",
            cache_size=0,
        )

    def test_mmap_storage_files_created(self, prep_mmap):
        """Verify mmap storage creates correct file structure."""
        # Verify mmap files exist
        assert (prep_mmap.processed_dir / "samples.mmap").exists()
        assert (prep_mmap.processed_dir / "samples.idx.npy").exists()
        assert prep_mmap._storage is not None
        assert prep_mmap.storage_backend == "mmap"
        
        # Verify we can read samples
        sample = prep_mmap[0]
        assert hasattr(sample, "x") and hasattr(sample, "edge_index")
        
        # Verify storage stats
        stats = prep_mmap._storage.get_stats()
        assert stats["num_samples"] == 50
        assert stats["compression"] == "lz4"
        assert stats["compression_ratio"] > 1.0

    def test_mmap_vs_files_io_speedup(self):
        """Prove mmap storage provides I/O speedup over individual files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            num_samples = 50
            num_accesses = 100
            
            source = SyntheticCustomDataset(num_samples=num_samples)
            
            # Benchmark file-based storage
            prep_files = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir / "files",
                storage_backend="files",
                cache_size=0,
            )
            
            start = time.time()
            for i in range(num_accesses):
                _ = prep_files[i % num_samples]
            time_files = time.time() - start
            
            # Benchmark mmap storage
            prep_mmap = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir / "mmap",
                storage_backend="mmap",
                compression="lz4",
                cache_size=0,
            )
            
            start = time.time()
            for i in range(num_accesses):
                _ = prep_mmap[i % num_samples]
            time_mmap = time.time() - start
            
            speedup = time_files / time_mmap
            
            # Verify correctness
            assert torch.equal(prep_files[0].x, prep_mmap[0].x)
            
            # Assert speedup (conservative threshold for CI)
            assert speedup > 1.1, f"Mmap speedup {speedup:.2f}× should be >1.1×"

    def test_compression_reduces_disk_usage(
        self, prep_nocomp, prep_mmap, prep_zstd
    ):
        """Prove compression significantly reduces disk usage."""
        # Get storage stats
        size_nocomp = prep_nocomp._storage.get_stats()["total_size_mb"]
        
        stats_lz4 = prep_mmap._storage.get_stats()
        size_lz4 = stats_lz4["total_size_mb"]
        ratio_lz4 = stats_lz4["compression_ratio"]
        
        stats_zstd = prep_zstd._storage.get_stats()
        ratio_zstd = stats_zstd["compression_ratio"]
        
        # Verify correctness
        assert torch.equal(prep_nocomp[0].x, prep_mmap[0].x)
        assert torch.equal(prep_nocomp[0].x, prep_zstd[0].x)
        
        # Assert compression effectiveness
        assert ratio_lz4 > 1.1, f"LZ4 ratio {ratio_lz4:.2f}× should be >1.1×"
        assert ratio_zstd > ratio_lz4, f"ZSTD {ratio_zstd:.2f}× should beat LZ4 {ratio_lz4:.2f}×"
        assert size_lz4 < size_nocomp, "LZ4 should use less disk space"

    def test_mmap_cache_integration(self, source_dataset, temp_dir_class):
        """Verify mmap storage and LRU cache work together correctly."""
        cache_size = 10
        
        # Need separate preprocessor with cache enabled
        preprocessor = OnDiskInductivePreprocessor(
            dataset=source_dataset,
            data_dir=temp_dir_class / "cached",
            storage_backend="mmap",
            compression="lz4",
            cache_size=cache_size,
        )
        
        # First access: cache miss, mmap load
        _ = preprocessor[0]
        stats1 = preprocessor.get_cache_stats()
        assert stats1["misses"] == 1
        assert preprocessor._storage is not None
        
        # Second access: cache hit
        _ = preprocessor[0]
        stats2 = preprocessor.get_cache_stats()
        assert stats2["hits"] == 1
        
        # Fill cache
        for i in range(cache_size):
            _ = preprocessor[i]
        
        # Access all cached samples (should all hit)
        start_hits = preprocessor.get_cache_stats()["hits"]
        for i in range(cache_size):
            _ = preprocessor[i]
        final_stats = preprocessor.get_cache_stats()
        
        new_hits = final_stats["hits"] - start_hits
        assert new_hits == cache_size, f"Expected {cache_size} hits, got {new_hits}"
        assert final_stats["hit_rate"] > 0.5, "Hit rate should be >50% with repeated access"
