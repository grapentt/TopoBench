"""Tests for OGBN-Papers100M on-disk dataset."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from topobench.data.datasets.ogbn_papers100m_ondisk import (
    OGBNPapers100MOnDiskDataset,
)


class TestOGBNPapers100MOnDiskDataset:
    """Tests for OGBNPapers100MOnDiskDataset."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock Papers100M dataset for testing."""
        # Create directory structure
        dataset_root = tmp_path / "ogbn_papers100M"
        processed_dir = dataset_root / "processed"
        processed_dir.mkdir(parents=True)

        # Create mock data (small scale for testing)
        num_nodes = 1000
        num_edges = 5000
        num_features = 128

        # Mock edge_index
        edge_index = np.random.randint(0, num_nodes, (2, num_edges), dtype=np.int64)
        np.save(processed_dir / "edge_index.npy", edge_index)

        # Mock node features
        node_feat = np.random.randn(num_nodes, num_features).astype(np.float32)
        np.save(processed_dir / "node_feat.npy", node_feat)

        # Mock labels
        node_label = np.random.randint(0, 10, num_nodes, dtype=np.int64)
        np.save(processed_dir / "node_label.npy", node_label)

        # Mock metadata
        with open(processed_dir / "num_nodes.txt", "w") as f:
            f.write(str(num_nodes))

        return tmp_path

    def test_dataset_initialization(self, mock_dataset_dir):
        """Test dataset can be initialized."""
        dataset = OGBNPapers100MOnDiskDataset(
            root=mock_dataset_dir,
            num_samples=100,
            k_hop=1,
            cache_samples=False,  # Disable caching for test speed
        )

        assert len(dataset) == 100
        assert dataset.num_nodes_total == 1000

    def test_sample_generation(self, mock_dataset_dir):
        """Test that samples can be generated."""
        dataset = OGBNPapers100MOnDiskDataset(
            root=mock_dataset_dir,
            num_samples=10,
            k_hop=0,  # Just node itself for speed
            cache_samples=False,
        )

        # Generate a sample
        sample = dataset[0]

        assert isinstance(sample, Data)
        assert sample.x is not None
        assert sample.y is not None
        assert sample.edge_index is not None

    def test_lightweight_pickling(self, mock_dataset_dir):
        """Test that dataset pickle is small."""
        dataset = OGBNPapers100MOnDiskDataset(
            root=mock_dataset_dir,
            num_samples=100,
            k_hop=1,
            cache_samples=False,
        )

        # Pickle the dataset
        pickled = pickle.dumps(dataset)
        pickle_size_kb = len(pickled) / 1024

        # Should be very small (< 50KB even with memmaps)
        assert (
            pickle_size_kb < 50
        ), f"Pickle too large: {pickle_size_kb:.2f} KB"

        # Verify unpickling works
        dataset_restored = pickle.loads(pickled)
        assert len(dataset_restored) == len(dataset)

    def test_memory_mapped_loading(self, mock_dataset_dir):
        """Test that memory-mapped arrays don't load into RAM."""
        dataset = OGBNPapers100MOnDiskDataset(
            root=mock_dataset_dir,
            num_samples=100,
            k_hop=1,
            cache_samples=False,
        )

        # Check that arrays are memory-mapped
        assert isinstance(dataset.edge_index_mmap, np.memmap)
        assert isinstance(dataset.node_feat_mmap, np.memmap)
        assert isinstance(dataset.node_label_mmap, np.memmap)

    def test_deterministic_generation(self, mock_dataset_dir):
        """Test that generation is deterministic with same seed."""
        dataset1 = OGBNPapers100MOnDiskDataset(
            root=mock_dataset_dir,
            num_samples=10,
            k_hop=0,
            cache_samples=False,
            seed=42,
        )

        dataset2 = OGBNPapers100MOnDiskDataset(
            root=mock_dataset_dir,
            num_samples=10,
            k_hop=0,
            cache_samples=False,
            seed=42,
        )

        # Same seed should give same samples
        for i in range(5):
            sample1 = dataset1[i]
            sample2 = dataset2[i]
            assert torch.allclose(sample1.x, sample2.x)
            assert torch.allclose(sample1.y, sample2.y)

    def test_properties(self, mock_dataset_dir):
        """Test dataset properties."""
        dataset = OGBNPapers100MOnDiskDataset(
            root=mock_dataset_dir,
            num_samples=50,
            k_hop=1,
            cache_samples=False,
        )

        assert dataset.num_features == 128
        assert isinstance(dataset.num_classes, int)
        assert dataset.num_classes > 0

    def test_repr(self, mock_dataset_dir):
        """Test string representation."""
        dataset = OGBNPapers100MOnDiskDataset(
            root=mock_dataset_dir,
            num_samples=100,
            k_hop=2,
        )

        repr_str = repr(dataset)
        assert "OGBNPapers100MOnDiskDataset" in repr_str
        assert "total_nodes=1,000" in repr_str
        assert "num_samples=100" in repr_str
        assert "k_hop=2" in repr_str


@pytest.mark.slow
@pytest.mark.skipif(
    True, reason="Skipped by default - downloads 44GB dataset"
)
class TestOGBNPapers100MOnDiskDatasetIntegration:
    """Integration tests with real data (slow, marked for optional runs)."""

    def test_real_download_and_process(self):
        """Test downloading and processing real Papers100M data.

        WARNING: This downloads ~44GB of data and requires ~150GB processing space.
        Only run explicitly for validation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = OGBNPapers100MOnDiskDataset(
                root=tmpdir,
                num_samples=1000,  # Just 1K samples for test
                k_hop=2,
            )

            # Verify dataset loaded
            assert len(dataset) == 1000
            assert dataset.num_nodes_total == 111059956  # Papers100M size

            # Verify sample structure
            sample = dataset[0]
            assert isinstance(sample, Data)
            assert sample.x.shape[1] == 128  # Feature dimension
            assert sample.num_nodes > 0
