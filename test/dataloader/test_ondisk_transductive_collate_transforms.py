"""Test transform application in OnDiskTransductiveCollate."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

from topobench.data.preprocessor import OnDiskTransductivePreprocessor
from topobench.dataloader import OnDiskTransductiveCollate


class TestOnDiskTransductiveCollateTransforms:
    """Test suite for transform application in transductive collate function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def small_graph(self):
        """Create a small test graph."""
        # Create Erdős-Rényi graph with 20 nodes
        edge_index = erdos_renyi_graph(num_nodes=20, edge_prob=0.3)
        
        data = Data(
            x=torch.randn(20, 8),  # 8-dim features
            edge_index=edge_index,
            y=torch.randint(0, 3, (20,)),  # 3 classes
            train_mask=torch.zeros(20, dtype=torch.bool),
            val_mask=torch.zeros(20, dtype=torch.bool),
            test_mask=torch.zeros(20, dtype=torch.bool),
        )
        
        # Set splits
        data.train_mask[:12] = True
        data.val_mask[12:16] = True
        data.test_mask[16:] = True
        
        return data

    @pytest.fixture
    def transforms_config(self):
        """Create transform configuration."""
        return OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2,
            }
        })

    def test_collate_without_transforms(self, small_graph, temp_dir):
        """Test collate function without transforms (baseline)."""
        # Create preprocessor WITHOUT transforms
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph,
            data_dir=Path(temp_dir) / "no_transforms",
            transforms_config=None,
            max_structure_size=3,
        )
        preprocessor.build_index()
        
        # Create collate function
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        # Query batch
        node_ids = [0, 1, 2, 3, 4]
        batch = collate_fn([node_ids])
        
        # Basic assertions
        assert batch.num_nodes == len(node_ids)
        assert batch.x.shape == (len(node_ids), 8)
        assert hasattr(batch, 'edge_index')
        
        # Should NOT have simplicial complex structures
        # (transforms create individual attributes like x_0, x_1, etc.)
        assert not hasattr(batch, 'x_0'), "Should not have x_0 without transforms"
        assert not hasattr(batch, 'hodge_laplacian_0'), "Should not have laplacians without transforms"
        assert not hasattr(batch, 'incidence_1'), "Should not have incidences without transforms"

    def test_collate_with_transforms(self, small_graph, temp_dir, transforms_config):
        """Test collate function WITH transforms."""
        # Create preprocessor WITH transforms
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph,
            data_dir=Path(temp_dir) / "with_transforms",
            transforms_config=transforms_config,
            max_structure_size=3,
        )
        preprocessor.build_index()
        
        # Create collate function
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        # Verify transform is initialized
        assert collate_fn.transform is not None, "Transform should be initialized"
        
        # Query batch
        node_ids = [0, 1, 2, 3, 4]
        batch = collate_fn([node_ids])
        
        # Basic assertions
        assert batch.num_nodes == len(node_ids)
        assert batch.x.shape == (len(node_ids), 8)
        assert hasattr(batch, 'edge_index')
        
        # SHOULD have simplicial complex structures after transform
        # (transforms create individual attributes: x_0, x_1, etc.)
        assert hasattr(batch, 'x_0'), "Should have x_0 after transforms"
        assert hasattr(batch, 'x_1'), "Should have x_1 after transforms"
        assert hasattr(batch, 'incidence_1'), "Should have incidence_1 after transforms"
        
        # Verify laplacians exist
        assert hasattr(batch, 'hodge_laplacian_0') or hasattr(batch, 'down_laplacian_1'), \
            "Should have laplacian matrices after transforms"
        
        # Verify dimensions
        assert batch.x_0.shape[0] == len(node_ids), "x_0 should have correct number of nodes"
        assert batch.x_0.shape[1] == 8, "x_0 should preserve feature dimension"

    def test_multiple_batches_with_transforms(self, small_graph, temp_dir, transforms_config):
        """Test that transforms work correctly across multiple batches."""
        # Create preprocessor
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph,
            data_dir=Path(temp_dir) / "multi_batch",
            transforms_config=transforms_config,
            max_structure_size=3,
        )
        preprocessor.build_index()
        
        # Create collate function
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        # Query multiple batches
        batch1 = collate_fn([[0, 1, 2]])
        batch2 = collate_fn([[3, 4, 5]])
        batch3 = collate_fn([[0, 5, 10]])
        
        # All batches should have simplicial complex structures
        for i, batch in enumerate([batch1, batch2, batch3], 1):
            assert hasattr(batch, 'x_0'), f"Batch {i} should have x_0"
            assert hasattr(batch, 'x_1'), f"Batch {i} should have x_1"
            assert hasattr(batch, 'incidence_1'), f"Batch {i} should have incidence_1"

    def test_transform_preserves_features(self, small_graph, temp_dir, transforms_config):
        """Test that transforms preserve node features."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph,
            data_dir=Path(temp_dir) / "preserve_features",
            transforms_config=transforms_config,
            max_structure_size=3,
        )
        preprocessor.build_index()
        
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        # Query batch
        node_ids = [0, 1, 2]
        batch = collate_fn([node_ids])
        
        # Original features should be preserved in x_0
        assert hasattr(batch, 'x_0')
        original_features = batch.x_0
        
        # Should match the original features from graph
        expected_features = small_graph.x[node_ids]
        assert torch.allclose(original_features, expected_features), \
            "Node features should be preserved after transform"

    def test_transform_with_empty_structures(self, temp_dir, transforms_config):
        """Test transform application when no structures are found."""
        # Create graph with no triangles (tree structure)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2],
            [1, 0, 2, 1, 3]
        ], dtype=torch.long)
        
        data = Data(
            x=torch.randn(4, 8),
            edge_index=edge_index,
            y=torch.randint(0, 2, (4,)),
        )
        
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=data,
            data_dir=Path(temp_dir) / "no_structures",
            transforms_config=transforms_config,
            max_structure_size=3,
        )
        preprocessor.build_index()
        
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        # Query batch
        batch = collate_fn([[0, 1, 2]])
        
        # Transform should still work even with no triangles
        assert hasattr(batch, 'x_0'), "Should have x_0 even without structures"
        assert hasattr(batch, 'x_1'), "Should have x_1 even without structures"
        assert hasattr(batch, 'incidence_1'), "Should have incidence_1"

    def test_transform_config_types(self, small_graph, temp_dir):
        """Test different transform configuration formats."""
        # Test with nested "liftings" key (Hydra format)
        nested_config = OmegaConf.create({
            "liftings": {
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                }
            }
        })
        
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph,
            data_dir=Path(temp_dir) / "nested_config",
            transforms_config=nested_config,
            max_structure_size=3,
        )
        preprocessor.build_index()
        
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        # Should handle nested config
        assert collate_fn.transform is not None
        
        # Query batch
        batch = collate_fn([[0, 1, 2]])
        assert hasattr(batch, 'x_0'), "Should work with nested config"
        assert hasattr(batch, 'x_1'), "Should have x_1 with nested config"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
