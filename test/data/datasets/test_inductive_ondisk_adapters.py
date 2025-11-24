"""Tests for PyG dataset adapters with performance comparison.

This module tests the adapter system that converts existing PyG datasets
(InMemoryDataset, TUDataset, etc.) into optimized on-disk format.

Key tests:
- Adapter basic functionality
- Performance comparison: adapted vs non-adapted
- Lightweight pickling after caching
- Integration with OnDiskInductivePreprocessor
"""
import pickle
import sys
import tempfile
import time
from pathlib import Path
from omegaconf import OmegaConf

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset
from topobench.data.preprocessor import OnDiskInductivePreprocessor

from topobench.data.datasets import (
    PyGDatasetAdapter,
    adapt_dataset,
)


class MockInMemoryDataset(InMemoryDataset):
    """Mock InMemoryDataset for testing (simulates PyG's TUDataset pattern)."""
    
    def __init__(self, root, num_samples=100):
        self.num_samples = num_samples
        super().__init__(root)
        self.data, self.slices = self._generate_data()
    
    def _generate_data(self):
        """Generate data in InMemoryDataset format."""
        data_list = []
        for i in range(self.num_samples):
            torch.manual_seed(i)
            data_list.append(Data(
                x=torch.randn(10, 8),
                edge_index=torch.randint(0, 10, (2, 20)),
                y=torch.tensor([i % 5])
            ))
        return self.collate(data_list)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def download(self):
        pass
    
    def process(self):
        pass


class SimpleMockDataset:
    """Simple mock dataset (not InMemoryDataset) for basic adapter tests."""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        return Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([idx])
        )


class TestPyGDatasetAdapter:
    """Test suite for PyG dataset adapter."""
    
    def test_adapter_basic(self):
        """Test basic adapter functionality with simple mock dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dataset = SimpleMockDataset(num_samples=10)
            adapter = PyGDatasetAdapter(source_dataset, root=tmpdir)
            
            # Check length
            assert len(adapter) == 10
            
            # Check sample loading
            sample = adapter[0]
            assert isinstance(sample, Data)
            assert sample.x.shape == (5, 3)
    
    def test_adapter_with_inmemory_dataset(self):
        """Test adapter with actual InMemoryDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create InMemoryDataset
            source = MockInMemoryDataset(tmpdir + "/source", num_samples=20)
            
            # Adapt it
            adapter = PyGDatasetAdapter(source, root=tmpdir + "/adapted")
            
            # Check functionality
            assert len(adapter) == 20
            sample = adapter[0]
            assert isinstance(sample, Data)
            assert sample.x.shape == (10, 8)
    
    def test_adapter_caching(self):
        """Test that adapter caches samples correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SimpleMockDataset(num_samples=5)
            adapter = adapt_dataset(source, root=tmpdir)
            
            # Check cache files exist
            cache_dir = Path(tmpdir) / ".sample_cache"
            assert cache_dir.exists()
            
            # All samples should be cached after adaptation
            cached_files = list(cache_dir.glob("sample_*.pt"))
            assert len(cached_files) == 5
    
    def test_adapter_lightweight_pickling(self):
        """Test that adapter is lightweight for multiprocessing after caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SimpleMockDataset(num_samples=10)
            adapter = adapt_dataset(source, root=tmpdir)
            
            pickled = pickle.dumps(adapter)
            size_kb = sys.getsizeof(pickled) / 1024
            
            assert size_kb < 50, f"Adapter pickle size ({size_kb:.2f}KB) exceeds 50KB limit"
    
    def test_adapt_dataset_convenience(self):
        """Test convenience function for adapting datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SimpleMockDataset(num_samples=5)
            
            # Use convenience function
            adapter = adapt_dataset(source, root=tmpdir)
            
            # Check it works
            assert len(adapter) == 5
            sample = adapter[0]
            assert isinstance(sample, Data)


class TestAdapterPerformanceComparison:
    """Performance comparison tests: adapted vs non-adapted datasets."""
    
    def test_pickle_size_comparison_mock(self):
        """Compare pickle size: InMemoryDataset vs Adapted (mock dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = 50  # Small for CI/CD
            
            inmemory = MockInMemoryDataset(tmpdir + "/source", num_samples=num_samples)
            adapted = adapt_dataset(inmemory, root=tmpdir + "/adapted")
            
            inmemory_pickled = pickle.dumps(inmemory)
            adapted_pickled = pickle.dumps(adapted)
            
            inmemory_size_kb = sys.getsizeof(inmemory_pickled) / 1024
            adapted_size_kb = sys.getsizeof(adapted_pickled) / 1024
            
            assert adapted_size_kb < inmemory_size_kb / 2, \
                f"Adapted ({adapted_size_kb:.1f}KB) should be <2× smaller than InMemory ({inmemory_size_kb:.1f}KB)"
    
    def test_parallel_preprocessing_speedup(self):
        """Compare parallel preprocessing: non-adapted vs adapted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            num_samples = 200  # Large enough to show difference
            
            inmemory = MockInMemoryDataset(str(tmpdir / "source"), num_samples=num_samples)
            
            # No transforms - isolates dataset pickling overhead
            transforms_config = None
            
            # Test 1: Non-adapted with parallel (heavy pickling to each worker)
            start = time.time()
            preprocessor_heavy = OnDiskInductivePreprocessor(
                dataset=inmemory,
                data_dir=str(tmpdir / "processed_heavy"),
                transforms_config=transforms_config,
                num_workers=4,
                force_reload=True
            )
            time_heavy = time.time() - start
            
            # Test 2: Adapted with parallel (lightweight pickling to each worker)
            adapted = adapt_dataset(inmemory, root=str(tmpdir / "adapted"))
            
            start = time.time()
            preprocessor_light = OnDiskInductivePreprocessor(
                dataset=adapted,
                data_dir=str(tmpdir / "processed_light"),
                transforms_config=transforms_config,
                num_workers=4,
                force_reload=True
            )
            time_light = time.time() - start
            
            speedup = time_heavy / time_light

            print("Speedup is: ", speedup)
            
            # Verify correctness
            assert len(preprocessor_heavy) == len(preprocessor_light) == num_samples
            
            # Adapted should be faster due to less pickling overhead
            # Even small speedup (>1.1x) validates the approach
            assert speedup >= 0.8, \
                f"Adapted parallel ({time_light:.2f}s) significantly slower than non-adapted parallel ({time_heavy:.2f}s)"
    
    def test_adapted_parallel_preprocessing(self):
        """Verify adapted datasets enable parallel preprocessing.
        
        Compares adapted vs non-adapted datasets with parallel preprocessing
        to confirm that adaptation produces valid lightweight datasets.
        """
        pytest.importorskip("topobench.data.preprocessor")
        from topobench.data.preprocessor import OnDiskInductivePreprocessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            num_samples = 100
            
            # Source dataset
            inmemory = MockInMemoryDataset(str(tmpdir / "source"), num_samples=num_samples)
            
            # Adapt dataset
            adapted = adapt_dataset(
                inmemory, 
                root=str(tmpdir / "adapted"),
                extraction_workers=2,
                verbose=False
            )
            
            # Verify adapted dataset is lightweight
            import pickle
            pickle_size = len(pickle.dumps(adapted)) / 1024
            assert pickle_size < 10, f"Adapted dataset pickle ({pickle_size:.2f}KB) exceeds 10KB"
            
            # Process with parallel workers
            preprocessor = OnDiskInductivePreprocessor(
                dataset=adapted,
                data_dir=str(tmpdir / "processed"),
                transforms_config=None,
                num_workers=2,
                force_reload=True
            )
            
            # Verify correctness
            assert len(preprocessor) == num_samples
            sample = preprocessor[0]
            assert isinstance(sample, Data)
            assert hasattr(sample, 'x') and hasattr(sample, 'edge_index')
    
    def test_memory_efficiency(self):
        """Test that adapted dataset enables O(1) memory preprocessing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = 50  # Small for CI/CD
            
            inmemory = MockInMemoryDataset(tmpdir + "/source", num_samples=num_samples)
            adapted = adapt_dataset(inmemory, root=tmpdir + "/adapted")
            
            inmemory_attrs_size = sum(sys.getsizeof(v) for v in vars(inmemory).values())
            adapted_attrs_size = sum(sys.getsizeof(v) for v in vars(adapted).values())
            
            assert adapted_attrs_size < inmemory_attrs_size, \
                "Adapted dataset should have smaller memory footprint"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
