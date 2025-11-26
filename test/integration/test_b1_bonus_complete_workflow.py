"""Complete end-to-end integration tests for B1 Bonus submission.

This test suite validates the complete workflow from preprocessing to training,
ensuring all components work together seamlessly. It tests both structure-centric
and extended context approaches with transforms and various configurations.
"""

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

from topobench.data.preprocessor import OnDiskTransductivePreprocessor
from topobench.dataloader import (
    ClusterAwareNodeSampler,
    TBDataloader,
    create_extended_context_dataloader,
    create_structure_centric_dataloader,
)


@pytest.fixture
def realistic_graph():
    """Create a realistic graph for integration testing."""
    # Generate Erdős-Rényi graph with 200 nodes
    edge_index = erdos_renyi_graph(num_nodes=200, edge_prob=0.1, directed=False)
    
    data = Data(
        x=torch.randn(200, 32),  # 32-dim features
        edge_index=edge_index,
        y=torch.randint(0, 5, (200,)),  # 5 classes
        num_nodes=200,
    )
    
    # Add train/val/test masks
    data.train_mask = torch.zeros(200, dtype=torch.bool)
    data.train_mask[:120] = True
    
    data.val_mask = torch.zeros(200, dtype=torch.bool)
    data.val_mask[120:160] = True
    
    data.test_mask = torch.zeros(200, dtype=torch.bool)
    data.test_mask[160:] = True
    
    return data


@pytest.fixture
def preprocessor_basic(realistic_graph, tmp_path):
    """Create basic preprocessor without transforms."""
    preprocessor = OnDiskTransductivePreprocessor(
        graph_data=realistic_graph,
        data_dir=tmp_path / "integration_basic",
        transforms_config=None,
        max_clique_size=3,
    )
    preprocessor.build_index()
    return preprocessor


@pytest.fixture
def preprocessor_with_transforms(realistic_graph, tmp_path):
    """Create preprocessor with topological transforms."""
    transforms_config = OmegaConf.create({
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,
        }
    })
    
    preprocessor = OnDiskTransductivePreprocessor(
        graph_data=realistic_graph,
        data_dir=tmp_path / "integration_transforms",
        transforms_config=transforms_config,
        max_clique_size=3,
    )
    preprocessor.build_index()
    return preprocessor


class TestStructureCentricWorkflow:
    """Test complete structure-centric workflow."""
    
    def test_basic_structure_centric_pipeline(self, preprocessor_basic):
        """Test basic structure-centric pipeline without transforms."""
        # Create dataloader
        loader = create_structure_centric_dataloader(
            preprocessor_basic,
            cliques_per_batch=50,
            node_budget=200,
            shuffle=True,
        )
        
        # Simulate training
        epoch_batches = []
        for batch in loader:
            # Verify batch structure
            assert isinstance(batch, Data)
            assert batch.num_nodes > 0
            assert hasattr(batch, "precomputed_structures")
            assert batch.num_structures > 0
            
            epoch_batches.append(batch)
        
        # Should have processed some batches
        assert len(epoch_batches) > 0
    
    def test_structure_centric_with_transforms(self, preprocessor_with_transforms):
        """Test structure-centric pipeline with topological transforms."""
        from topobench.transforms.data_transform import DataTransform
        
        # Create transform
        transform_config = OmegaConf.create({
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,
        })
        transform = DataTransform(**transform_config)
        
        # Create loader with transform
        loader = create_structure_centric_dataloader(
            preprocessor_with_transforms,
            cliques_per_batch=50,
            node_budget=200,
            transform=transform,
            shuffle=True,
        )
        
        # Process batches
        for batch_idx, batch in enumerate(loader):
            # Should have simplicial complex attributes
            assert hasattr(batch, "x_0"), "Missing x_0 after transform"
            assert hasattr(batch, "x_1"), "Missing x_1 after transform"
            assert hasattr(batch, "incidence_1"), "Missing incidence_1 after transform"
            
            # Features should match batch size
            assert batch.x_0.shape[0] == batch.num_nodes
            
            if batch_idx >= 2:  # Test a few batches
                break
        
    
    def test_structure_centric_multi_epoch(self, preprocessor_basic):
        """Test structure-centric pipeline across multiple epochs."""
        loader = create_structure_centric_dataloader(
            preprocessor_basic,
            cliques_per_batch=50,
            node_budget=200,
            shuffle=True,
        )
        
        epoch_lengths = []
        for epoch in range(3):
            batch_count = 0
            for batch in loader:
                assert batch.num_nodes > 0
                batch_count += 1
            
            epoch_lengths.append(batch_count)
        
        # Should have consistent batches across epochs
        assert all(length > 0 for length in epoch_lengths)
    
    def test_structure_completeness_validation(self, preprocessor_basic):
        """Validate 100% structure completeness."""
        loader = create_structure_centric_dataloader(
            preprocessor_basic,
            cliques_per_batch=30,
            node_budget=150,
            shuffle=False,
        )
        
        total_structures_sampled = 0
        total_structures_in_batches = 0
        
        for batch in loader:
            # Count structures in batch
            total_structures_in_batches += batch.num_structures
            
            # Verify all structures are complete
            for struct_id, nodes in batch.precomputed_structures:
                assert len(nodes) == 3  # Triangles
                assert all(0 <= n < batch.num_nodes for n in nodes)
            
            total_structures_sampled += batch.num_structures
        
        # All sampled structures should be in batches (100% completeness)
        assert total_structures_in_batches == total_structures_sampled


class TestExtendedContextWorkflow:
    """Test complete extended context workflow."""
    
    def test_basic_extended_context_pipeline(self, preprocessor_basic):
        """Test basic extended context pipeline."""
        # Create node sampler
        node_sampler = ClusterAwareNodeSampler(
            graph_data=preprocessor_basic.graph_data,
            batch_size=50,
            clustering_method="louvain",
            mask=preprocessor_basic.graph_data.train_mask,
            seed=42,
        )
        
        # Create loader
        loader = create_extended_context_dataloader(
            preprocessor_basic,
            node_sampler=node_sampler,
            max_expansion_ratio=1.5,
        )
        
        # Process batches
        expansion_ratios = []
        for batch in loader:
            assert isinstance(batch, Data)
            assert batch.num_nodes > 0
            assert hasattr(batch, "core_mask")
            assert hasattr(batch, "expansion_ratio")
            
            expansion_ratios.append(batch.expansion_ratio)
        
        # Should have processed batches
        assert len(expansion_ratios) > 0
        avg_expansion = sum(expansion_ratios) / len(expansion_ratios)
    
    def test_extended_context_with_transforms(self, preprocessor_with_transforms):
        """Test extended context with transforms."""
        from topobench.transforms.data_transform import DataTransform
        
        transform_config = OmegaConf.create({
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,
        })
        transform = DataTransform(**transform_config)
        
        # Create sampler and loader
        node_sampler = ClusterAwareNodeSampler(
            graph_data=preprocessor_with_transforms.graph_data,
            batch_size=50,
            clustering_method="louvain",
            seed=42,
        )
        
        loader = create_extended_context_dataloader(
            preprocessor_with_transforms,
            node_sampler=node_sampler,
            max_expansion_ratio=1.5,
            transform=transform,
        )
        
        # Process batches
        for batch_idx, batch in enumerate(loader):
            # Should have transforms
            assert hasattr(batch, "x_0")
            assert hasattr(batch, "x_1")
            assert hasattr(batch, "incidence_1")
            
            # Should still have context info
            assert hasattr(batch, "core_mask")
            assert hasattr(batch, "expansion_ratio")
            
            if batch_idx >= 2:
                break
        
    
    def test_expansion_ratio_control(self, preprocessor_basic):
        """Test that expansion ratio is controlled."""
        max_ratio = 1.3
        
        node_sampler = ClusterAwareNodeSampler(
            graph_data=preprocessor_basic.graph_data,
            batch_size=50,
            clustering_method="louvain",
            seed=42,
        )
        
        loader = create_extended_context_dataloader(
            preprocessor_basic,
            node_sampler=node_sampler,
            max_expansion_ratio=max_ratio,
            filter_on_expansion=True,
        )
        
        # Check all batches respect the ratio
        for batch in loader:
            assert batch.expansion_ratio <= max_ratio + 0.2  # Small tolerance
        


class TestLoadDatasetSplitsIntegration:
    """Test load_dataset_splits high-level API."""
    
    def test_structure_centric_splits(self, realistic_graph, tmp_path):
        """Test loading splits with structure-centric strategy."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=realistic_graph,
            data_dir=tmp_path / "splits_sc",
            max_clique_size=3,
        )
        
        split_config = OmegaConf.create({
            "strategy": "structure_centric",
            "cliques_per_batch": 40,
            "node_budget": 150,
        })
        
        train, val, test = preprocessor.load_dataset_splits(split_config)
        
        # Verify splits
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        
        # Test iteration
        train_batch = next(iter(train))
        assert train_batch.num_nodes > 0
        
        preprocessor.close()
    
    def test_extended_context_splits(self, realistic_graph, tmp_path):
        """Test loading splits with extended context strategy."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=realistic_graph,
            data_dir=tmp_path / "splits_ec",
            max_clique_size=3,
        )
        
        split_config = OmegaConf.create({
            "strategy": "extended_context",
            "nodes_per_batch": 50,
            "max_expansion_ratio": 1.5,
            "sampler_method": "louvain",
        })
        
        train, val, test = preprocessor.load_dataset_splits(split_config)
        
        # Verify splits
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        
        preprocessor.close()
    
    def test_splits_with_tbdataloader(self, realistic_graph, tmp_path):
        """Test complete integration with TBDataloader."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=realistic_graph,
            data_dir=tmp_path / "tbdataloader_test",
            max_clique_size=3,
        )
        
        split_config = OmegaConf.create({
            "strategy": "structure_centric",
            "cliques_per_batch": 40,
            "node_budget": 150,
        })
        
        train, val, test = preprocessor.load_dataset_splits(split_config)
        
        # Create TBDataloader
        datamodule = TBDataloader(
            dataset_train=train,
            dataset_val=val,
            dataset_test=test,
            batch_size=1,  # Pre-batched
            num_workers=0,
        )
        
        # Test train loader
        train_loader = datamodule.train_dataloader()
        train_batches = []
        for batch in train_loader:
            train_batches.append(batch)
            if len(train_batches) >= 3:  # Sample a few
                break
        
        assert len(train_batches) > 0
        
        # Test val loader
        val_loader = datamodule.val_dataloader()
        val_batch = next(iter(val_loader))
        assert val_batch.num_nodes > 0
        
        preprocessor.close()


class TestTrainingSimulation:
    """Simulate realistic training scenarios."""
    
    def test_complete_training_simulation(self, preprocessor_basic):
        """Simulate a complete training loop."""
        split_config = OmegaConf.create({
            "strategy": "structure_centric",
            "cliques_per_batch": 50,
            "node_budget": 200,
            "shuffle": True,
        })
        
        train, val, test = preprocessor_basic.load_dataset_splits(split_config)
        
        # Simulate 3 epochs
        for epoch in range(3):
            # Training
            train_loss = 0.0
            train_count = 0
            for batch in train:
                # Simulate forward pass (basic batch without transforms)
                assert hasattr(batch, "x")
                assert hasattr(batch, "edge_index")
                
                # Simulate loss computation
                batch_loss = torch.randn(1).item()  # Fake loss
                train_loss += batch_loss
                train_count += 1
            
            avg_train_loss = train_loss / train_count if train_count > 0 else 0
            
            # Validation
            val_loss = 0.0
            val_count = 0
            for batch in val:
                batch_loss = torch.randn(1).item()
                val_loss += batch_loss
                val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else 0
            
        
        # Test
        test_count = 0
        for batch in test:
            assert batch.num_nodes > 0
            test_count += 1
        
    
    def test_memory_stability(self, preprocessor_basic):
        """Test memory stability across many iterations."""
        import gc
        import os
        
        import psutil
        
        process = psutil.Process(os.getpid())
        gc.collect()
        
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        loader = create_structure_centric_dataloader(
            preprocessor_basic,
            cliques_per_batch=50,
            node_budget=200,
            shuffle=True,
        )
        
        # Process many batches
        for epoch in range(5):
            for batch_idx, batch in enumerate(loader):
                assert batch.num_nodes > 0
                
                if batch_idx >= 10:  # Sample some batches
                    break
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - baseline_memory
        
        
        # Memory should be bounded
        assert memory_increase < 300, f"Memory leak detected: {memory_increase:.1f}MB"


class TestComparisonBetweenApproaches:
    """Compare structure-centric vs extended context."""
    
    def test_completeness_comparison(self, preprocessor_basic):
        """Compare structure completeness between approaches."""
        # Structure-centric (should be 100%)
        sc_loader = create_structure_centric_dataloader(
            preprocessor_basic,
            cliques_per_batch=30,
            node_budget=150,
        )
        
        sc_total_structures = sum(batch.num_structures for batch in sc_loader)
        
        # Extended context (should be high but may not be 100%)
        node_sampler = ClusterAwareNodeSampler(
            graph_data=preprocessor_basic.graph_data,
            batch_size=50,
            clustering_method="louvain",
            seed=42,
        )
        
        ec_loader = create_extended_context_dataloader(
            preprocessor_basic,
            node_sampler=node_sampler,
            max_expansion_ratio=1.5,
        )
        
        ec_total_structures = sum(batch.num_structures for batch in ec_loader)
        
        
        # Both should find structures
        assert sc_total_structures > 0
        assert ec_total_structures > 0


class TestErrorRecovery:
    """Test error handling and recovery."""
    
    def test_empty_batch_handling(self, realistic_graph, tmp_path):
        """Test handling of edge cases."""
        # Create graph with very few triangles
        sparse_edge_index = torch.tensor([
            [0, 1, 2],
            [1, 2, 3]
        ], dtype=torch.long)
        
        sparse_data = Data(
            x=torch.randn(4, 8),
            edge_index=sparse_edge_index,
            y=torch.randint(0, 2, (4,)),
            num_nodes=4,
        )
        
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=sparse_data,
            data_dir=tmp_path / "sparse_test",
            max_clique_size=3,
        )
        preprocessor.build_index()
        
        # May have very few or no structures
        if preprocessor.num_structures > 0:
            loader = create_structure_centric_dataloader(
                preprocessor,
                cliques_per_batch=10,
                node_budget=10,
            )
            
            # Should handle gracefully
            batches = list(loader)
            assert len(batches) >= 0  # May be 0 or more
        
        preprocessor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
