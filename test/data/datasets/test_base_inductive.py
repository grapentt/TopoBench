"""Tests for base inductive dataset classes.

Tests the high-performance base classes that enable 2.29-5× parallel speedup:
- BaseOnDiskInductiveDataset
- FileBasedInductiveDataset
- OnDemandInductiveDataset
"""

import pickle
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from topobench.data.datasets import (
    FileBasedInductiveDataset,
    OnDemandInductiveDataset,
)


class SimpleGeneratedDataset(OnDemandInductiveDataset):
    """Simple generated dataset for testing."""
    
    def _generate_sample(self, idx, rng):
        """Generate a simple test sample."""
        x = torch.randn(10, 5, generator=rng)
        edge_index = torch.randint(0, 10, (2, 15), generator=rng)
        y = torch.tensor([idx % 3])
        return Data(x=x, edge_index=edge_index, y=y)


class SimpleFileDataset(FileBasedInductiveDataset):
    """Simple file-based dataset for testing."""
    
    def _load_file(self, file_path):
        """Load a PyTorch file."""
        return torch.load(file_path, weights_only=False)


class TestOnDemandInductiveDataset:
    """Test suite for OnDemandInductiveDataset."""
    
    def test_generated_dataset_basic(self):
        """Test basic functionality of generated dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=20)
            
            # Check length
            assert len(dataset) == 20
            
            # Check sample generation
            sample = dataset[0]
            assert isinstance(sample, Data)
            assert sample.x.shape == (10, 5)
            assert sample.edge_index.shape == (2, 15)
            assert sample.y.item() == 0
            
            # Check another sample
            sample_10 = dataset[10]
            assert sample_10.y.item() == 10 % 3
    
    def test_generated_dataset_caching(self):
        """Test that caching works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=10)
            
            # First access - generates and caches
            sample1 = dataset[0]
            
            # Check cache file exists
            cache_file = Path(tmpdir) / ".sample_cache" / "sample_000000.pt"
            assert cache_file.exists()
            
            # Second access - loads from cache
            sample2 = dataset[0]
            
            # Samples should be identical
            assert torch.allclose(sample1.x, sample2.x)
            assert torch.allclose(sample1.edge_index.float(), sample2.edge_index.float())
    
    def test_generated_dataset_pickling(self):
        """Test pickling correctness and lightweight size for parallel processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset1 = SimpleGeneratedDataset(tmpdir, num_samples=100)
            
            # Access one sample to populate cache
            sample1 = dataset1[5]
            
            # Pickle and unpickle
            pickled = pickle.dumps(dataset1)
            dataset2 = pickle.loads(pickled)
            
            # Verify correctness
            assert len(dataset2) == 100
            sample2 = dataset2[5]
            
            # Verify pickle is lightweight for parallel processing (< 10KB)
            size_kb = sys.getsizeof(pickled) / 1024
            assert size_kb < 10, f"Pickle size ({size_kb:.2f}KB) exceeds 10KB limit for parallel efficiency"
            
            # Samples should be identical (same seed)
            assert torch.allclose(sample1.x, sample2.x)
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic across instances."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                # Create two datasets with same seed
                dataset1 = SimpleGeneratedDataset(tmpdir1, num_samples=10, seed=42)
                dataset2 = SimpleGeneratedDataset(tmpdir2, num_samples=10, seed=42)
                
                # Generate samples from both
                for idx in [0, 5, 9]:
                    sample1 = dataset1[idx]
                    sample2 = dataset2[idx]
                    
                    # Should be identical
                    assert torch.allclose(sample1.x, sample2.x)
                    assert torch.equal(sample1.edge_index, sample2.edge_index)
                    assert sample1.y.item() == sample2.y.item()


class TestFileBasedInductiveDataset:
    """Test suite for FileBasedInductiveDataset."""
    
    def test_file_based_dataset(self):
        """Test basic file-based dataset functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create some test files
            for i in range(10):
                data = Data(
                    x=torch.randn(8, 4),
                    edge_index=torch.randint(0, 8, (2, 12)),
                    y=torch.tensor([i])
                )
                torch.save(data, tmpdir / f"sample_{i:03d}.pt")
            
            # Create dataset
            dataset = SimpleFileDataset(tmpdir, file_pattern="sample_*.pt")
            
            # Check length
            assert len(dataset) == 10
            
            # Check samples
            for i in range(10):
                sample = dataset[i]
                assert isinstance(sample, Data)
                assert sample.y.item() == i
    
    def test_file_based_dataset_custom_pattern(self):
        """Test file-based dataset with custom glob pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create files with different patterns
            for i in range(5):
                data = Data(x=torch.randn(5, 3), edge_index=torch.randint(0, 5, (2, 8)))
                torch.save(data, tmpdir / f"graph_{i}.pt")
                torch.save(data, tmpdir / f"other_{i}.txt")  # Different extension
            
            # Dataset should only find .pt files matching pattern
            dataset = SimpleFileDataset(tmpdir, file_pattern="graph_*.pt")
            assert len(dataset) == 5
    
    def test_file_based_dataset_pickling(self):
        """Test pickling correctness and lightweight size for parallel processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create 100 files
            for i in range(100):
                data = Data(x=torch.randn(10, 5), edge_index=torch.randint(0, 10, (2, 15)))
                torch.save(data, tmpdir / f"sample_{i:04d}.pt")
            
            # Create dataset
            dataset = SimpleFileDataset(tmpdir, file_pattern="sample_*.pt")
            assert len(dataset) == 100
            
            # Pickle and verify lightweight for parallel processing
            pickled = pickle.dumps(dataset)
            size_kb = sys.getsizeof(pickled) / 1024
            assert size_kb < 10, f"Pickle size ({size_kb:.2f}KB) exceeds 10KB limit for parallel efficiency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
