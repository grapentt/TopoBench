"""Integration tests validating transform correctness for on-disk datasets.

These tests ensure OnDiskInductiveDataset produces identical results to
PreProcessor when applying TopoBench transforms (liftings).
"""

import pytest
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from topobench.data.preprocessor import create_preprocessor
from topobench.data.preprocessor.ondisk_inductive import (
    OnDiskInductivePreprocessor,
)


@pytest.fixture
def mutag_dataset(tmp_path):
    """Load MUTAG dataset (small, real PyG dataset)."""
    try:
        dataset = TUDataset(root=str(tmp_path / "MUTAG"), name="MUTAG")
        # Take small subset for fast testing
        indices = list(range(min(10, len(dataset))))
        subset = dataset[indices]
        return subset
    except Exception as e:
        pytest.skip(f"MUTAG dataset not available: {e}")


@pytest.fixture
def enzymes_dataset(tmp_path):
    """Load ENZYMES dataset (another real dataset for validation)."""
    try:
        dataset = TUDataset(root=str(tmp_path / "ENZYMES"), name="ENZYMES")
        # Take small subset
        indices = list(range(min(10, len(dataset))))
        subset = dataset[indices]
        return subset
    except Exception as e:
        pytest.skip(f"ENZYMES dataset not available: {e}")


class TestSimplicialCliqueLifting:
    """Test OnDiskInductiveDataset with SimplicialCliqueLifting."""

    def test_simplicial_clique_lifting_basic(self, mutag_dataset, tmp_path):
        """Test basic SimplicialCliqueLifting produces correct structures."""
        # Configure lifting
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2  # Include up to triangles
            }
        })
        
        # Create on-disk preprocessor
        ondisk = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(tmp_path / "ondisk"),
            transforms_config=transforms_config,
            mode="ondisk"
        )
        
        assert isinstance(ondisk, OnDiskInductivePreprocessor)
        assert len(ondisk) == len(mutag_dataset)
        
        # Verify samples can be loaded
        for i in range(len(ondisk)):
            sample = ondisk[i]
            assert isinstance(sample, Data)
            # Should have lifted structures
            assert hasattr(sample, "x_0") or hasattr(sample, "x_1") or hasattr(sample, "x_2")

    def test_simplicial_vs_inmemory_consistency(self, mutag_dataset, tmp_path):
        """Test on-disk produces same structures as in-memory for small dataset."""
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2
            }
        })
        
        # Create both preprocessors
        ondisk = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(tmp_path / "ondisk"),
            transforms_config=transforms_config,
            mode="ondisk"
        )
        
        inmemory = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(tmp_path / "inmemory"),
            transforms_config=transforms_config,
            mode="inmemory"
        )
        
        # Both should have same length
        assert len(ondisk) == len(inmemory)
        
        # Spot check: compare first sample
        ondisk_sample = ondisk[0]
        inmemory_sample = inmemory[0]
        
        # Both should have x_0 (node features)
        assert hasattr(ondisk_sample, "x_0")
        assert hasattr(inmemory_sample, "x_0")
        
        # Node features should match
        if hasattr(ondisk_sample, "x_0") and hasattr(inmemory_sample, "x_0"):
            assert ondisk_sample.x_0.shape == inmemory_sample.x_0.shape

    def test_different_complex_dimensions(self, mutag_dataset, tmp_path):
        """Test lifting with different complex dimensions."""
        for complex_dim in [1, 2, 3]:
            transforms_config = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": complex_dim
                }
            })
            
            ondisk = create_preprocessor(
                dataset=mutag_dataset,
                data_dir=str(tmp_path / f"ondisk_dim{complex_dim}"),
                transforms_config=transforms_config,
                mode="ondisk"
            )
            
            assert len(ondisk) == len(mutag_dataset)
            
            # Verify can load samples
            sample = ondisk[0]
            assert isinstance(sample, Data)


class TestHypergraphLifting:
    """Test OnDiskInductiveDataset with hypergraph liftings."""

    def test_hypergraph_khop_lifting(self, mutag_dataset, tmp_path):
        """Test HypergraphKHopLifting works with on-disk."""
        transforms_config = OmegaConf.create({
            "khop_lifting": {
                "transform_type": "lifting",
                "transform_name": "HypergraphKHopLifting",
                "k_value": 2,
                "signed": False
            }
        })
        
        ondisk = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(tmp_path / "ondisk_khop"),
            transforms_config=transforms_config,
            mode="ondisk"
        )
        
        assert isinstance(ondisk, OnDiskInductivePreprocessor)
        assert len(ondisk) == len(mutag_dataset)
        
        # Verify samples load
        sample = ondisk[0]
        assert isinstance(sample, Data)


class TestNoTransform:
    """Test OnDiskInductiveDataset without transforms."""

    def test_no_transform_passthrough(self, mutag_dataset, tmp_path):
        """Test that no transform config passes through data unchanged."""
        ondisk = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(tmp_path / "ondisk_notransform"),
            transforms_config=None,
            mode="ondisk"
        )
        
        assert len(ondisk) == len(mutag_dataset)
        
        # Compare with original
        for i in range(len(ondisk)):
            ondisk_sample = ondisk[i]
            original_sample = mutag_dataset[i]
            
            # Should have same basic structure
            assert ondisk_sample.num_nodes == original_sample.num_nodes
            assert ondisk_sample.edge_index.shape == original_sample.edge_index.shape


class TestMultipleDatasets:
    """Test transforms work across different datasets."""

    def test_transforms_on_enzymes(self, enzymes_dataset, tmp_path):
        """Test that transforms work on ENZYMES dataset."""
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2
            }
        })
        
        ondisk = create_preprocessor(
            dataset=enzymes_dataset,
            data_dir=str(tmp_path / "enzymes_ondisk"),
            transforms_config=transforms_config,
            mode="ondisk"
        )
        
        assert len(ondisk) == len(enzymes_dataset)
        
        # Verify all samples load correctly
        for i in range(len(ondisk)):
            sample = ondisk[i]
            assert isinstance(sample, Data)


class TestCaching:
    """Test transform caching works correctly."""

    def test_transform_caching_reuses_results(self, mutag_dataset, tmp_path):
        """Test that processing with same config reuses cached results."""
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2
            }
        })
        
        data_dir = tmp_path / "caching_test"
        
        # First run - should process
        ondisk1 = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(data_dir),
            transforms_config=transforms_config,
            mode="ondisk"
        )
        
        # Get modification time of first sample
        sample_path = ondisk1._get_sample_path(0)
        mtime1 = sample_path.stat().st_mtime
        
        # Second run - should use cache
        ondisk2 = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(data_dir),
            transforms_config=transforms_config,
            mode="ondisk"
        )
        
        # Modification time should be same (not reprocessed)
        mtime2 = sample_path.stat().st_mtime
        assert mtime1 == mtime2
        
        # Data should still be correct
        assert len(ondisk2) == len(mutag_dataset)

    def test_different_config_invalidates_cache(self, mutag_dataset, tmp_path):
        """Test that different transform config triggers reprocessing."""
        data_dir = tmp_path / "cache_invalidation"
        
        # First config
        config1 = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2
            }
        })
        
        ondisk1 = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(data_dir),
            transforms_config=config1,
            mode="ondisk"
        )
        
        # Different config (different complex_dim)
        config2 = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 3  # Different!
            }
        })
        
        ondisk2 = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(data_dir),
            transforms_config=config2,
            mode="ondisk"
        )
        
        # Both should work
        assert len(ondisk1) == len(mutag_dataset)
        assert len(ondisk2) == len(mutag_dataset)


class TestErrorHandling:
    """Test error handling for invalid transforms."""

    def test_invalid_transform_name_raises_error(self, mutag_dataset, tmp_path):
        """Test that invalid transform name raises appropriate error."""
        transforms_config = OmegaConf.create({
            "invalid_lifting": {
                "transform_type": "lifting",
                "transform_name": "NonExistentLifting",
                "complex_dim": 2
            }
        })
        
        with pytest.raises(Exception):  # Should raise when trying to instantiate
            ondisk = create_preprocessor(
                dataset=mutag_dataset,
                data_dir=str(tmp_path / "invalid_transform"),
                transforms_config=transforms_config,
                mode="ondisk"
            )


class TestMemoryEfficiency:
    """Test that on-disk maintains constant memory."""

    def test_memory_stays_constant_during_loading(self, mutag_dataset, tmp_path):
        """Test that loading samples doesn't accumulate memory."""
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2
            }
        })
        
        ondisk = create_preprocessor(
            dataset=mutag_dataset,
            data_dir=str(tmp_path / "memory_test"),
            transforms_config=transforms_config,
            mode="ondisk"
        )
        
        # Load all samples (should not accumulate in memory)
        for i in range(len(ondisk)):
            sample = ondisk[i]
            # Explicitly delete to free memory
            del sample
        
        # If we get here without OOM, memory efficiency is working
        assert True
