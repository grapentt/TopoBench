"""Tests for US County Demographics on-disk dataset."""

import os
import pickle
import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from topobench.data.datasets.us_county_demos_ondisk import (
    USCountyDemosOnDiskDataset,
)


class TestUSCountyDemosOnDiskDataset:
    """Tests for USCountyDemosOnDiskDataset."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def dataset_params(self):
        """Create dataset parameters."""
        return DictConfig({"year": 2012, "task_variable": "Election"})

    @pytest.fixture
    def mock_dataset_dir(self, temp_dir):
        """Create a mock dataset directory with minimal data.

        This fixture creates the necessary directory structure and a dummy
        processed file to avoid downloading real data during tests.
        """
        root = temp_dir
        name = "US-county-demos-test"
        year = 2012
        task_variable = "Election"

        # Create directory structure
        dataset_root = os.path.join(root, name)
        raw_dir = os.path.join(dataset_root, "raw")
        processed_root = os.path.join(
            dataset_root, f"{year}_{task_variable}"
        )
        processed_dir = os.path.join(processed_root, "processed")

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Create a minimal mock data object
        mock_data = Data(
            x=torch.randn(100, 7),  # 100 nodes, 7 features
            y=torch.randn(100),  # 100 labels
            edge_index=torch.randint(0, 100, (2, 500)),  # 500 edges
        )

        # Save mock data
        torch.save(mock_data, os.path.join(processed_dir, "data_0.pt"))

        # Create dummy raw files to pass existence check
        open(os.path.join(raw_dir, "county_graph.csv"), "w").close()
        open(os.path.join(raw_dir, f"county_stats_{year}.csv"), "w").close()

        return root, name, DictConfig({"year": year, "task_variable": task_variable})

    def test_ondisk_dataset_basic(self, mock_dataset_dir):
        """Test basic functionality of on-disk dataset."""
        root, name, params = mock_dataset_dir

        dataset = USCountyDemosOnDiskDataset(
            root=root, name=name, parameters=params
        )

        # Check dataset length
        assert len(dataset) == 1, "Dataset should contain 1 graph"

        # Check data loading
        data = dataset[0]
        assert isinstance(data, Data), "Should return PyG Data object"
        assert data.x is not None, "Data should have node features"
        assert data.y is not None, "Data should have labels"
        assert data.edge_index is not None, "Data should have edges"

    def test_ondisk_dataset_properties(self, mock_dataset_dir):
        """Test dataset properties."""
        root, name, params = mock_dataset_dir

        dataset = USCountyDemosOnDiskDataset(
            root=root, name=name, parameters=params
        )

        # Test properties
        assert dataset.num_features == 7, "Should have 7 features"
        assert dataset.num_nodes == 100, "Should have 100 nodes"
        assert isinstance(dataset.num_edges, int), "Should return edge count"

    def test_ondisk_dataset_caching(self, mock_dataset_dir):
        """Test that caching works correctly."""
        root, name, params = mock_dataset_dir

        dataset = USCountyDemosOnDiskDataset(
            root=root, name=name, parameters=params, cache_samples=True
        )

        # First access - should create cache
        data1 = dataset[0]

        # Check cache was created (cache is in processed_dir/.sample_cache)
        cache_dir = Path(dataset.root) / ".sample_cache"
        cache_files = list(cache_dir.glob("*.pt"))
        assert len(cache_files) > 0, "Cache file should be created"

        # Second access - should use cache
        data2 = dataset[0]

        # Data should be identical
        assert torch.allclose(data1.x, data2.x), "Cached data should match"
        assert torch.allclose(data1.y, data2.y), "Cached labels should match"

    def test_ondisk_dataset_pickling(self, mock_dataset_dir):
        """Test that dataset can be pickled efficiently."""
        root, name, params = mock_dataset_dir

        dataset = USCountyDemosOnDiskDataset(
            root=root, name=name, parameters=params
        )

        # Pickle the dataset
        pickled = pickle.dumps(dataset)
        pickle_size_kb = len(pickled) / 1024

        # Should be lightweight (< 100KB for small dataset)
        assert (
            pickle_size_kb < 100
        ), f"Pickled dataset should be small, got {pickle_size_kb:.2f}KB"

        # Unpickle and verify
        dataset_restored = pickle.loads(pickled)
        assert len(dataset_restored) == len(
            dataset
        ), "Restored dataset should have same length"

        # Verify data can still be accessed
        data_orig = dataset[0]
        data_restored = dataset_restored[0]
        assert torch.allclose(
            data_orig.x, data_restored.x
        ), "Restored data should match"

    def test_ondisk_dataset_repr(self, mock_dataset_dir):
        """Test string representation."""
        root, name, params = mock_dataset_dir

        dataset = USCountyDemosOnDiskDataset(
            root=root, name=name, parameters=params
        )

        repr_str = repr(dataset)
        assert "USCountyDemosOnDiskDataset" in repr_str
        assert name in repr_str
        assert str(params.year) in repr_str
        assert params.task_variable in repr_str

    def test_ondisk_dataset_multiple_access(self, mock_dataset_dir):
        """Test multiple accesses return consistent data."""
        root, name, params = mock_dataset_dir

        dataset = USCountyDemosOnDiskDataset(
            root=root, name=name, parameters=params
        )

        # Access same sample multiple times
        data1 = dataset[0]
        data2 = dataset[0]
        data3 = dataset[0]

        # All should be identical
        assert torch.allclose(data1.x, data2.x), "Data should be consistent"
        assert torch.allclose(data2.x, data3.x), "Data should be consistent"
        assert torch.allclose(data1.y, data2.y), "Labels should be consistent"
        assert torch.allclose(data2.y, data3.y), "Labels should be consistent"


@pytest.mark.slow
class TestUSCountyDemosOnDiskDatasetIntegration:
    """Integration tests that download real data (slow, marked for optional runs)."""

    def test_real_download_and_process(self):
        """Test downloading and processing real data.

        This test is marked as slow and should only be run explicitly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            params = DictConfig({"year": 2012, "task_variable": "Election"})

            dataset = USCountyDemosOnDiskDataset(
                root=tmpdir, name="US-county-demos", parameters=params
            )

            # Verify dataset is loaded
            assert len(dataset) > 0, "Should load at least one graph"

            # Verify data structure
            data = dataset[0]
            assert isinstance(data, Data)
            assert data.x is not None
            assert data.y is not None
            assert data.edge_index is not None

            # Verify dimensions make sense
            assert data.x.shape[0] > 0, "Should have nodes"
            assert data.x.shape[1] > 0, "Should have features"
