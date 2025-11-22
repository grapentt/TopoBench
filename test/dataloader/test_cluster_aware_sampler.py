"""Tests for ClusterAwareNodeSampler and HybridNodeSampler."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

from topobench.dataloader import ClusterAwareNodeSampler, HybridNodeSampler


@pytest.fixture
def small_graph():
    """Create a small test graph."""
    edge_index = erdos_renyi_graph(num_nodes=50, edge_prob=0.15, directed=False)
    data = Data(
        x=torch.randn(50, 8),
        edge_index=edge_index,
        y=torch.randint(0, 3, (50,)),
    )
    return data


@pytest.fixture
def graph_with_mask(small_graph):
    """Add train/val/test masks to graph."""
    small_graph.train_mask = torch.zeros(50, dtype=torch.bool)
    small_graph.train_mask[:30] = True
    small_graph.val_mask = torch.zeros(50, dtype=torch.bool)
    small_graph.val_mask[30:40] = True
    small_graph.test_mask = torch.zeros(50, dtype=torch.bool)
    small_graph.test_mask[40:] = True
    return small_graph


class TestClusterAwareNodeSampler:
    """Test ClusterAwareNodeSampler functionality."""
    
    def test_louvain_sampling(self, small_graph):
        """Test basic Louvain clustering sampling."""
        sampler = ClusterAwareNodeSampler(
            graph_data=small_graph,
            batch_size=10,
            clustering_method="louvain",
            shuffle=False,
            seed=42,
        )
        
        # Collect all batches
        batches = list(sampler)
        
        # Should have batches
        assert len(batches) > 0
        
        # Each batch should have nodes
        for batch in batches:
            assert len(batch) > 0
            assert len(batch) <= 10
        
        # All nodes should be covered (approximately)
        all_nodes = set()
        for batch in batches:
            all_nodes.update(batch)
        assert len(all_nodes) == 50
    
    def test_random_clustering(self, small_graph):
        """Test random clustering (baseline)."""
        sampler = ClusterAwareNodeSampler(
            graph_data=small_graph,
            batch_size=10,
            clustering_method="random",
            num_clusters=5,
            shuffle=False,
            seed=42,
        )
        
        batches = list(sampler)
        assert len(batches) > 0
        
        # Should create specified number of clusters
        assert len(sampler.clusters) == 5
    
    def test_with_mask(self, graph_with_mask):
        """Test sampling with train/val/test mask."""
        sampler = ClusterAwareNodeSampler(
            graph_data=graph_with_mask,
            batch_size=10,
            clustering_method="louvain",
            mask=graph_with_mask.train_mask,
            seed=42,
        )
        
        # Collect all batches
        batches = list(sampler)
        
        # All sampled nodes should be in train set
        for batch in batches:
            for node in batch:
                assert graph_with_mask.train_mask[node], \
                    f"Node {node} not in train mask"
    
    def test_shuffle(self, small_graph):
        """Test that shuffle produces different order."""
        sampler1 = ClusterAwareNodeSampler(
            graph_data=small_graph,
            batch_size=10,
            clustering_method="random",
            num_clusters=3,
            shuffle=True,
            seed=42,
        )
        
        sampler2 = ClusterAwareNodeSampler(
            graph_data=small_graph,
            batch_size=10,
            clustering_method="random",
            num_clusters=3,
            shuffle=True,
            seed=43,  # Different seed
        )
        
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        
        # Should have same number of batches
        assert len(batches1) == len(batches2)
        
        # First batches should likely be different (due to shuffle)
        assert batches1[0] != batches2[0]
    
    def test_batch_size_respected(self, small_graph):
        """Test that batch size is approximately respected."""
        batch_size = 10
        sampler = ClusterAwareNodeSampler(
            graph_data=small_graph,
            batch_size=batch_size,
            clustering_method="random",
            num_clusters=3,
            seed=42,
        )
        
        batches = list(sampler)
        
        # Most batches should be close to batch_size
        for batch in batches[:-1]:  # Exclude last batch (may be smaller)
            assert len(batch) <= batch_size


class TestHybridNodeSampler:
    """Test HybridNodeSampler functionality."""
    
    def test_cluster_strategy(self, small_graph):
        """Test pure cluster strategy."""
        sampler = HybridNodeSampler(
            graph_data=small_graph,
            batch_size=10,
            strategy="cluster",
            clustering_method="louvain",
            seed=42,
        )
        
        batches = list(sampler)
        assert len(batches) > 0
        
        # Should cover all nodes
        all_nodes = set()
        for batch in batches:
            all_nodes.update(batch)
        assert len(all_nodes) == 50
    
    def test_random_strategy(self, small_graph):
        """Test pure random strategy."""
        sampler = HybridNodeSampler(
            graph_data=small_graph,
            batch_size=10,
            strategy="random",
            seed=42,
        )
        
        batches = list(sampler)
        assert len(batches) > 0
    
    def test_hybrid_strategy(self, small_graph):
        """Test hybrid strategy (mix of cluster and random)."""
        sampler = HybridNodeSampler(
            graph_data=small_graph,
            batch_size=10,
            strategy="hybrid",
            cluster_ratio=0.7,
            clustering_method="louvain",
            seed=42,
        )
        
        batches = list(sampler)
        assert len(batches) > 0
    
    def test_invalid_strategy(self, small_graph):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            HybridNodeSampler(
                graph_data=small_graph,
                batch_size=10,
                strategy="invalid",
            )


class TestIntegrationWithCollate:
    """Test integration with OnDiskTransductiveCollate."""
    
    def test_cluster_sampling_with_collate(self, small_graph, tmp_path):
        """Test full pipeline: cluster sampling + collate + transforms."""
        from topobench.data.preprocessor import OnDiskTransductivePreprocessor
        from topobench.dataloader import OnDiskTransductiveCollate
        from omegaconf import OmegaConf
        
        # Create preprocessor with transforms
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2,
            }
        })
        
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph,
            data_dir=tmp_path / "index",
            transforms_config=transforms_config,
            max_structure_size=3,
        )
        preprocessor.build_index()
        
        # Create cluster-aware sampler
        sampler = ClusterAwareNodeSampler(
            graph_data=small_graph,
            batch_size=10,
            clustering_method="louvain",
            seed=42,
        )
        
        # Create collate function
        collate_fn = OnDiskTransductiveCollate(preprocessor, fully_contained=True)
        
        # Test sampling + collation
        for batch_nodes in sampler:
            batch = collate_fn([batch_nodes])
            
            # Verify batch has transform structures
            assert hasattr(batch, 'x_0'), "Should have x_0 after transform"
            assert hasattr(batch, 'x_1'), "Should have x_1 after transform"
            assert hasattr(batch, 'incidence_1'), "Should have incidence_1"
            
            # Verify batch size matches
            assert batch.num_nodes == len(batch_nodes)
            
            # Only test first batch
            break
        
        print("✓ Cluster sampling + collate + transforms working!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
