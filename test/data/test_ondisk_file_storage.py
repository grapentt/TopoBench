"""Test cases for OnDiskInductivePreprocessor file storage.

These tests would have caught the file storage bug where samples weren't being written.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from topobench.data.preprocessor.ondisk_inductive import OnDiskInductivePreprocessor
from topobench.transforms.data_transform import DataTransform


class SimpleMockDataset:
    """Simple dataset for testing."""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate simple graph."""
        return Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            y=torch.tensor([[1.0]]),
        )


def test_ondisk_files_are_written_sequential():
    """Test that sample files are actually written to disk (sequential processing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = SimpleMockDataset(num_samples=5)
        
        # Process with sequential (num_workers=1)
        preprocessor = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=None,  # No transforms
            num_workers=1,
            storage_backend="files",
        )
        
        # Check files exist
        assert len(preprocessor) == 5
        for i in range(5):
            sample_path = preprocessor.processed_dir / f"sample_{i:06d}.pt"
            assert sample_path.exists(), f"Sample {i} file not written!"
            
            # Verify file is loadable
            loaded = torch.load(sample_path, weights_only=False)
            assert isinstance(loaded, Data)
            assert loaded.x.shape == (5, 3)


def test_ondisk_files_are_written_parallel():
    """Test that sample files are written with parallel processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = SimpleMockDataset(num_samples=10)
        
        # Process with parallel workers
        preprocessor = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=None,
            num_workers=2,
            storage_backend="files",
        )
        
        # Check all files exist
        assert len(preprocessor) == 10
        for i in range(10):
            sample_path = preprocessor.processed_dir / f"sample_{i:06d}.pt"
            assert sample_path.exists(), f"Sample {i} file not written!"


def test_ondisk_with_transforms_files_written():
    """Test that files are written when transforms are applied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = SimpleMockDataset(num_samples=5)
        
        # Simple transform config
        transforms_config = {
            "identity": {
                "transform_type": "feature",
                "transform_name": "FeatureLiftProjection",
                "projection": "sum",
            }
        }
        
        preprocessor = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=transforms_config,
            num_workers=1,
            storage_backend="files",
        )
        
        # Verify files exist
        for i in range(5):
            sample_path = preprocessor.processed_dir / f"sample_{i:06d}.pt"
            assert sample_path.exists(), f"Sample {i} file not written with transforms!"


def test_ondisk_incremental_processing_each_transform_writes():
    """Test that incremental processing writes files for EACH transform in chain.
    
    This is the KEY test that would have caught the bug!
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = SimpleMockDataset(num_samples=5)
        
        # Two transforms in chain
        transforms_config = {
            "transform1": {
                "transform_type": "feature",
                "transform_name": "FeatureLiftProjection",
                "projection": "sum",
            },
            "transform2": {
                "transform_type": "feature", 
                "transform_name": "FeatureLiftProjection",
                "projection": "mean",
            }
        }
        
        preprocessor = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=transforms_config,
            num_workers=1,
            storage_backend="files",
        )
        
        # Check that transform chain was created
        assert preprocessor.transform_chain is not None
        assert len(preprocessor.transform_chain) >= 1  # At least one heavy transform
        
        # CRITICAL: Each transform in chain should have written files
        for chain_entry in preprocessor.transform_chain:
            output_dir = Path(chain_entry["output_dir"])
            
            # Check that THIS transform's directory has files
            for i in range(5):
                sample_path = output_dir / f"sample_{i:06d}.pt"
                assert sample_path.exists(), (
                    f"Transform {chain_entry['transform_id']} didn't write sample {i}! "
                    f"This is the bug we fixed!"
                )


def test_ondisk_processing_reports_failures():
    """Test that processing failures are reported correctly."""
    
    class FailingDataset:
        """Dataset that fails on certain indices."""
        
        def __len__(self):
            return 5
        
        def __getitem__(self, idx):
            if idx == 2:
                raise ValueError("Intentional failure")
            return Data(
                x=torch.randn(3, 2),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
            )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = FailingDataset()
        
        # This should not crash, but should report failures
        preprocessor = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=None,
            num_workers=1,
            storage_backend="files",
        )
        
        # Some samples should exist, some shouldn't
        assert (preprocessor.processed_dir / "sample_000000.pt").exists()
        assert (preprocessor.processed_dir / "sample_000001.pt").exists()
        assert not (preprocessor.processed_dir / "sample_000002.pt").exists()
        assert (preprocessor.processed_dir / "sample_000003.pt").exists()


def test_ondisk_mmap_storage_files_created():
    """Test that mmap storage creates proper files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = SimpleMockDataset(num_samples=5)
        
        preprocessor = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=None,
            num_workers=1,
            storage_backend="mmap",
        )
        
        # Check mmap files exist
        mmap_path = preprocessor.processed_dir / "samples.mmap"
        idx_path = preprocessor.processed_dir / "samples.idx.npy"
        metadata_path = preprocessor.processed_dir / "metadata.json"
        
        assert mmap_path.exists(), "Mmap data file not created!"
        assert idx_path.exists(), "Mmap index file not created!"
        assert metadata_path.exists(), "Mmap metadata file not created!"


def test_ondisk_can_load_samples_after_writing():
    """Test that written samples can be loaded via __getitem__."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = SimpleMockDataset(num_samples=5)
        
        preprocessor = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=None,
            num_workers=1,
            storage_backend="files",
        )
        
        # Load each sample via __getitem__
        for i in range(5):
            loaded = preprocessor[i]
            assert isinstance(loaded, Data)
            assert loaded.x.shape == (5, 3)
            assert loaded.edge_index.shape[0] == 2


def test_ondisk_force_reload_rewrites_files():
    """Test that force_reload actually rewrites files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = SimpleMockDataset(num_samples=3)
        
        # First processing
        preprocessor1 = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=None,
            num_workers=1,
            storage_backend="files",
        )
        
        sample_path = preprocessor1.processed_dir / "sample_000000.pt"
        first_mtime = sample_path.stat().st_mtime
        
        import time
        time.sleep(0.1)  # Ensure different timestamp
        
        # Second processing with force_reload
        preprocessor2 = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=tmpdir,
            transforms_config=None,
            num_workers=1,
            storage_backend="files",
            force_reload=True,
        )
        
        second_mtime = sample_path.stat().st_mtime
        assert second_mtime > first_mtime, "Files not rewritten with force_reload!"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
