"""Tests for OGBN-products loader."""

import pytest
import torch
from omegaconf import OmegaConf

from topobench.data.loaders import OGBNProductsLoader


@pytest.fixture
def loader_config(tmp_path):
    """Create configuration for OGBN-products loader."""
    return OmegaConf.create(
        {"data_dir": str(tmp_path / "ogbn_products"), "data_name": "ogbn-products"}
    )


class TestOGBNProductsLoader:
    """Test OGBN-products loader."""

    def test_loader_instantiation(self, loader_config):
        """Test that loader can be instantiated."""
        loader = OGBNProductsLoader(loader_config)
        assert loader is not None
        assert loader.name == "ogbn-products"

    @pytest.mark.slow
    def test_loader_loads_dataset(self, loader_config):
        """Test that loader loads dataset correctly.

        This test is marked as slow because it downloads the dataset
        (~1.5GB) on first run.
        """
        try:
            import ogb  # noqa: F401
        except ImportError:
            pytest.skip("ogb package not installed")

        loader = OGBNProductsLoader(loader_config)
        dataset, data_dir = loader.load()

        # Should be a single-graph dataset
        assert len(dataset) == 1

        # Get the graph
        graph = dataset[0]

        # Verify expected properties
        assert hasattr(graph, "x")
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "y")
        assert hasattr(graph, "train_mask")
        assert hasattr(graph, "val_mask")
        assert hasattr(graph, "test_mask")

        # Verify dimensions
        assert graph.x.shape[1] == 100  # 100-dimensional features
        assert graph.edge_index.shape[0] == 2  # [src, dst]

        # Verify masks sum to total nodes
        total_masked = (
            graph.train_mask.sum() + graph.val_mask.sum() + graph.test_mask.sum()
        )
        assert total_masked == graph.num_nodes

        # Verify node count (should be ~2.4M)
        assert graph.num_nodes > 2_000_000
        assert graph.num_nodes < 3_000_000

        # Verify edge count (should be ~61M)
        assert graph.edge_index.shape[1] > 60_000_000

    def test_loader_raises_without_ogb(self, loader_config, monkeypatch):
        """Test that loader raises ImportError if ogb not installed."""
        # Mock ogb import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "ogb" in name:
                raise ImportError("Mocked ogb import failure")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        loader = OGBNProductsLoader(loader_config)

        with pytest.raises(ImportError, match="ogb package is required"):
            loader.load_dataset()

    def test_loader_config_parameters(self, tmp_path):
        """Test that loader respects config parameters."""
        config = OmegaConf.create(
            {"data_dir": str(tmp_path / "custom_dir"), "data_name": "ogbn-products"}
        )

        loader = OGBNProductsLoader(config)
        assert str(loader.root_data_dir) == str(tmp_path / "custom_dir")
        assert loader.name == "ogbn-products"
