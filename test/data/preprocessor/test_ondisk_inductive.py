"""Tests for OnDiskInductiveDataset."""

import json
from pathlib import Path

import pytest
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from topobench.data.preprocessor.ondisk_inductive import (
    OnDiskInductiveDataset,
)


class TestOnDiskInductiveDataset:
    """Test suite for OnDiskInductiveDataset."""

    @pytest.fixture
    def sample_data_list(self):
        """Create small synthetic dataset."""
        data_list = []
        for i in range(10):
            # Create simple graph with node features
            x = torch.randn(5, 3)  # 5 nodes, 3 features
            edge_index = torch.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
            )
            y = torch.tensor([i % 3])  # Class label
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        return data_list

    @pytest.fixture
    def synthetic_dataset(self, sample_data_list):
        """Create synthetic PyG dataset."""

        class SyntheticDataset(torch_geometric.data.InMemoryDataset):
            def __init__(self, data_list):
                super().__init__(root=None)
                self.data_list = data_list
                self._data, self.slices = self.collate(data_list)

            def len(self):
                return len(self.data_list)

            def get(self, idx):
                return self.data_list[idx]

        return SyntheticDataset(sample_data_list)

    @pytest.fixture
    def mutag_dataset(self, tmp_path):
        """Load small MUTAG dataset for integration testing."""
        try:
            dataset = TUDataset(root=str(tmp_path), name="MUTAG")
            return dataset
        except Exception:
            pytest.skip("MUTAG dataset not available")

    def test_initialization_no_transforms(
        self, synthetic_dataset, tmp_path
    ):
        """Test dataset initialization without transforms."""
        data_dir = tmp_path / "test_no_transforms"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        assert len(dataset) == 10
        assert dataset.processed_dir.exists()
        assert dataset.metadata_path.exists()

    def test_initialization_with_force_reload(
        self, synthetic_dataset, tmp_path
    ):
        """Test force reload recreates processed data."""
        data_dir = tmp_path / "test_force_reload"

        # Create dataset first time
        dataset1 = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Modify a sample file to detect reload
        sample_path = dataset1._get_sample_path(0)
        original_data = torch.load(sample_path)

        # Corrupt the file
        torch.save({"corrupted": True}, sample_path)

        # Force reload should recreate
        dataset2 = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
            force_reload=True,
        )

        # Verify data is restored
        reloaded_data = torch.load(sample_path)
        assert "corrupted" not in reloaded_data
        assert "x" in reloaded_data or hasattr(reloaded_data, "x")

    def test_len(self, synthetic_dataset, tmp_path):
        """Test __len__ method returns correct size."""
        data_dir = tmp_path / "test_len"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        assert len(dataset) == len(synthetic_dataset)
        assert len(dataset) == 10

    def test_getitem_valid_index(self, synthetic_dataset, tmp_path):
        """Test __getitem__ with valid index."""
        data_dir = tmp_path / "test_getitem"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Get first sample
        sample = dataset[0]
        assert isinstance(sample, Data)
        assert hasattr(sample, "x")
        assert hasattr(sample, "edge_index")
        assert hasattr(sample, "y")

        # Verify data matches original
        original = synthetic_dataset[0]
        assert torch.equal(sample.x, original.x)
        assert torch.equal(sample.edge_index, original.edge_index)

    def test_getitem_all_indices(self, synthetic_dataset, tmp_path):
        """Test __getitem__ for all indices."""
        data_dir = tmp_path / "test_getitem_all"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Verify we can load all samples
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert isinstance(sample, Data)
            assert sample.y.item() == idx % 3  # Verify label

    def test_getitem_invalid_index_negative(
        self, synthetic_dataset, tmp_path
    ):
        """Test __getitem__ with negative index raises IndexError."""
        data_dir = tmp_path / "test_invalid_negative"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        with pytest.raises(IndexError):
            _ = dataset[-1]

    def test_getitem_invalid_index_too_large(
        self, synthetic_dataset, tmp_path
    ):
        """Test __getitem__ with out-of-range index."""
        data_dir = tmp_path / "test_invalid_large"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        with pytest.raises(IndexError):
            _ = dataset[100]

    def test_sequential_processing_files_exist(
        self, synthetic_dataset, tmp_path
    ):
        """Test that all sample files are created during processing."""
        data_dir = tmp_path / "test_sequential"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Verify all sample files exist
        for idx in range(len(dataset)):
            sample_path = dataset._get_sample_path(idx)
            assert sample_path.exists()
            assert sample_path.stat().st_size > 0

    def test_metadata_saved_correctly(self, synthetic_dataset, tmp_path):
        """Test metadata file is created with correct content."""
        data_dir = tmp_path / "test_metadata"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Verify metadata file exists
        assert dataset.metadata_path.exists()

        # Load and verify content
        with open(dataset.metadata_path) as f:
            metadata = json.load(f)

        assert "num_samples" in metadata
        assert metadata["num_samples"] == 10
        assert "source_dataset" in metadata
        assert "processed_dir" in metadata

    def test_caching_reuses_processed_data(
        self, synthetic_dataset, tmp_path
    ):
        """Test that cached data is reused on second initialization."""
        data_dir = tmp_path / "test_caching"

        # First initialization - processes data
        dataset1 = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Get modification time of first sample
        sample_path = dataset1._get_sample_path(0)
        mtime1 = sample_path.stat().st_mtime

        # Second initialization - should use cache
        dataset2 = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Verify file was not rewritten (same mtime)
        mtime2 = sample_path.stat().st_mtime
        assert mtime1 == mtime2

        # Verify data is still correct
        assert len(dataset2) == 10
        sample = dataset2[0]
        assert isinstance(sample, Data)

    def test_repr_method(self, synthetic_dataset, tmp_path):
        """Test __repr__ returns informative string."""
        data_dir = tmp_path / "test_repr"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        repr_str = repr(dataset)
        assert "OnDiskInductiveDataset" in repr_str
        assert "num_samples=10" in repr_str
        assert "processed_dir" in repr_str

    def test_processed_dir_structure(self, synthetic_dataset, tmp_path):
        """Test processed directory has correct structure."""
        data_dir = tmp_path / "test_structure"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Verify directory structure
        assert dataset.processed_dir.is_dir()
        assert dataset.metadata_path.is_file()

        # Count sample files
        sample_files = list(dataset.processed_dir.glob("sample_*.pt"))
        assert len(sample_files) == 10

    def test_sample_file_naming(self, synthetic_dataset, tmp_path):
        """Test sample files are named correctly with zero-padding."""
        data_dir = tmp_path / "test_naming"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Check specific file names
        assert (dataset.processed_dir / "sample_000000.pt").exists()
        assert (dataset.processed_dir / "sample_000009.pt").exists()

        # Verify naming pattern
        path = dataset._get_sample_path(5)
        assert path.name == "sample_000005.pt"

    def test_integration_with_mutag(self, mutag_dataset, tmp_path):
        """Integration test with real MUTAG dataset."""
        if mutag_dataset is None:
            pytest.skip("MUTAG dataset not available")

        data_dir = tmp_path / "test_mutag"

        # Process small subset
        dataset = OnDiskInductiveDataset(
            dataset=mutag_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        assert len(dataset) > 0
        sample = dataset[0]
        assert isinstance(sample, Data)

    def test_split_idx_preservation(self, tmp_path):
        """Test that split_idx is preserved from source dataset."""

        class DatasetWithSplits(torch_geometric.data.InMemoryDataset):
            def __init__(self):
                super().__init__(root=None)
                self.split_idx = {"train": [0, 1], "val": [2], "test": [3]}
                data_list = [Data(x=torch.randn(3, 2)) for _ in range(4)]
                self._data, self.slices = self.collate(data_list)

            def len(self):
                return 4

            def get(self, idx):
                return Data(x=torch.randn(3, 2))

        source = DatasetWithSplits()
        data_dir = tmp_path / "test_splits"

        dataset = OnDiskInductiveDataset(
            dataset=source,
            data_dir=data_dir,
            transforms_config=None,
        )

        assert hasattr(dataset, "split_idx")
        assert dataset.split_idx == source.split_idx

    def test_empty_dataset_handling(self, tmp_path):
        """Test handling of empty dataset."""

        class EmptyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError("Empty dataset")

        source = EmptyDataset()
        data_dir = tmp_path / "test_empty"

        dataset = OnDiskInductiveDataset(
            dataset=source,
            data_dir=data_dir,
            transforms_config=None,
        )

        assert len(dataset) == 0
        with pytest.raises(IndexError):
            _ = dataset[0]

    def test_missing_sample_file_raises_error(
        self, synthetic_dataset, tmp_path
    ):
        """Test that missing sample file raises helpful error."""
        data_dir = tmp_path / "test_missing"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Delete a sample file
        sample_path = dataset._get_sample_path(0)
        sample_path.unlink()

        # Attempt to load should raise FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            _ = dataset[0]

        assert "Sample file not found" in str(exc_info.value)
        assert "force_reload=True" in str(exc_info.value)

    def test_corrupted_metadata_triggers_reprocessing(
        self, synthetic_dataset, tmp_path
    ):
        """Test that corrupted metadata triggers reprocessing."""
        data_dir = tmp_path / "test_corrupted"

        # Create dataset first time
        dataset1 = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Corrupt metadata file
        with open(dataset1.metadata_path, "w") as f:
            f.write("{invalid json")

        # Should trigger reprocessing
        dataset2 = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Verify metadata is fixed
        assert dataset2.metadata_path.exists()
        with open(dataset2.metadata_path) as f:
            metadata = json.load(f)
        assert metadata["num_samples"] == 10

    def test_load_dataset_splits_inductive(
        self, synthetic_dataset, tmp_path
    ):
        """Test load_dataset_splits with inductive setting."""
        data_dir = tmp_path / "test_splits_inductive"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        # Create split parameters with required fields
        split_params = DictConfig(
            {
                "learning_setting": "inductive",
                "split_type": "random",
                "data_seed": 42,
                "train_prop": 0.6,
                "val_prop": 0.2,
                "data_split_dir": str(tmp_path / "splits"),
            }
        )

        # Load splits
        train_ds, val_ds, test_ds = dataset.load_dataset_splits(split_params)

        # Verify splits exist
        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None

        # Verify sizes roughly match proportions
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(dataset)

    def test_load_dataset_splits_wrong_setting_raises_error(
        self, synthetic_dataset, tmp_path
    ):
        """Test that transductive setting raises error."""
        data_dir = tmp_path / "test_splits_error"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        split_params = DictConfig({"learning_setting": "transductive"})

        with pytest.raises(ValueError) as exc_info:
            dataset.load_dataset_splits(split_params)

        assert "inductive" in str(exc_info.value).lower()

    def test_load_dataset_splits_missing_setting_raises_error(
        self, synthetic_dataset, tmp_path
    ):
        """Test that missing learning_setting raises error."""
        data_dir = tmp_path / "test_splits_missing"

        dataset = OnDiskInductiveDataset(
            dataset=synthetic_dataset,
            data_dir=data_dir,
            transforms_config=None,
        )

        split_params = DictConfig({})

        with pytest.raises(ValueError) as exc_info:
            dataset.load_dataset_splits(split_params)

        assert "No learning setting" in str(exc_info.value)
