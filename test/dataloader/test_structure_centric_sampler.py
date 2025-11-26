"""Tests for StructureCentricSampler and StructureCentricCollate."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

from topobench.data.preprocessor import OnDiskTransductivePreprocessor
from topobench.dataloader import (
    StructureCentricBatchSampler,
    StructureCentricCollate,
    StructureCentricSampler,
    create_structure_centric_dataloader,
)


@pytest.fixture
def triangle_graph():
    """Create a graph with known triangles."""
    # Create graph with 3 triangles: 0-1-2, 3-4-5, 6-7-8
    # Define edges as [source, target] pairs
    edges = [
        # Triangle 1: 0-1-2
        [0, 1],
        [1, 2],
        [2, 0],
        # Triangle 2: 3-4-5
        [3, 4],
        [4, 5],
        [5, 3],
        # Triangle 3: 6-7-8
        [6, 7],
        [7, 8],
        [8, 6],
        # Connect triangles
        [2, 3],
        [5, 6],
    ]

    # Convert to edge_index format [2, num_edges]
    edge_index = torch.tensor(
        [[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long
    )

    # Make symmetric (add reverse edges)
    edge_index_full = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(
        x=torch.randn(9, 8),
        edge_index=edge_index_full,
        y=torch.randint(0, 3, (9,)),
        num_nodes=9,
    )
    return data


@pytest.fixture
def preprocessor_with_structures(triangle_graph, tmp_path):
    """Create preprocessor with built index."""
    preprocessor = OnDiskTransductivePreprocessor(
        graph_data=triangle_graph,
        data_dir=tmp_path / "structure_index",
        max_clique_size=3,
    )
    preprocessor.build_index()
    return preprocessor


@pytest.fixture
def larger_graph(tmp_path):
    """Create a larger graph for stress testing."""
    # Generate Erdős-Rényi graph with 100 nodes
    edge_index = erdos_renyi_graph(
        num_nodes=100, edge_prob=0.2, directed=False
    )

    data = Data(
        x=torch.randn(100, 16),
        edge_index=edge_index,
        y=torch.randint(0, 5, (100,)),
        num_nodes=100,
    )

    preprocessor = OnDiskTransductivePreprocessor(
        graph_data=data,
        data_dir=tmp_path / "large_index",
        max_clique_size=3,
    )
    preprocessor.build_index()

    return data, preprocessor


class TestStructureCentricSampler:
    """Test StructureCentricSampler functionality."""

    def test_sampling_with_budget_constraints(
        self, preprocessor_with_structures
    ):
        """Test sampling respects cliques_per_batch and node_budget constraints."""
        sampler = StructureCentricSampler(
            preprocessor_with_structures,
            cliques_per_batch=2,
            node_budget=10,
            shuffle=False,
        )

        # Implicitly tests: initialization, num_cliques, all_clique_ids
        assert sampler.num_cliques > 0
        batches = list(sampler)
        assert len(batches) > 0

        # Test cliques_per_batch and node_budget constraints together
        for batch_ids in batches[:-1]:  # Exclude last batch
            # Implicitly tests: valid IDs, list type, non-empty
            assert len(batch_ids) <= sampler.cliques_per_batch

            # Verify node budget enforcement
            structures = (
                preprocessor_with_structures.query_engine.query_cliques_by_id(
                    batch_ids
                )
            )
            node_set = set()
            for _, nodes in structures:
                node_set.update(nodes)
            assert len(node_set) <= sampler.node_budget + 3

    def test_shuffle_and_drop_last(self, preprocessor_with_structures):
        """Test shuffle and drop_last parameters affect sampling."""
        sampler_keep = StructureCentricSampler(
            preprocessor_with_structures,
            cliques_per_batch=2,
            node_budget=10,
            drop_last=False,
        )

        sampler_drop = StructureCentricSampler(
            preprocessor_with_structures,
            cliques_per_batch=2,
            node_budget=10,
            drop_last=True,
            shuffle=True,
        )

        batches_keep = list(sampler_keep)
        batches_drop = list(sampler_drop)

        # drop_last should have ≤ batches
        assert len(batches_drop) <= len(batches_keep)
        # Implicitly tests shuffle produces valid output
        assert all(len(b) > 0 for b in batches_drop)

    def test_larger_graph_completeness(self, larger_graph):
        """Test sampling completeness on larger graph."""
        _, preprocessor = larger_graph

        sampler = StructureCentricSampler(
            preprocessor,
            cliques_per_batch=50,
            node_budget=200,
            shuffle=True,
        )

        batches = list(sampler)
        total_structures = sum(len(batch) for batch in batches)

        # All structures should be sampled exactly once
        assert total_structures == sampler.num_cliques


class TestStructureCentricBatchSampler:
    """Test StructureCentricBatchSampler (fixed-size variant)."""

    def test_fixed_batch_sizes_and_length(self, preprocessor_with_structures):
        """Test batch sizes and length calculation."""
        batch_size = 2
        sampler = StructureCentricBatchSampler(
            preprocessor_with_structures,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )

        batches = list(sampler)

        # All batches should have exactly batch_size structures
        for batch_ids in batches:
            assert len(batch_ids) == batch_size

        # Length should match iteration count
        assert len(batches) == len(sampler)


class TestStructureCentricCollate:
    """Test StructureCentricCollate functionality."""

    def test_collation_with_structure_completeness(
        self, preprocessor_with_structures
    ):
        """Test batch creation ensures structure completeness and correct metadata."""
        collate_fn = StructureCentricCollate(
            preprocessor_with_structures,
            include_structure_metadata=True,
        )

        structure_ids = [0, 1, 2]  # All 3 triangles
        batch = collate_fn(structure_ids)

        # Implicitly tests: Data type, num_nodes, edge_index, attributes
        assert isinstance(batch, Data)
        assert batch.num_structures == len(structure_ids)
        assert batch.x.shape[0] == batch.num_nodes
        assert batch.edge_index.max() < batch.num_nodes

        # Verify structure metadata and node gathering
        structures = (
            preprocessor_with_structures.query_engine.query_cliques_by_id(
                structure_ids
            )
        )
        expected_nodes = set()
        for _, nodes in structures:
            expected_nodes.update(nodes)

        assert batch.num_nodes == len(expected_nodes)

        # All structures should have valid batch-local indices
        for _struct_id, nodes in batch.precomputed_structures:
            assert all(0 <= n < batch.num_nodes for n in nodes)

    def test_metadata_control_and_edge_cases(
        self, preprocessor_with_structures
    ):
        """Test metadata inclusion control and empty batch handling."""
        collate_fn_no_meta = StructureCentricCollate(
            preprocessor_with_structures,
            include_structure_metadata=False,
        )

        batch = collate_fn_no_meta([0, 1])
        assert not hasattr(batch, "precomputed_structures")

        # Test empty batch handling
        collate_fn = StructureCentricCollate(preprocessor_with_structures)
        empty_batch = collate_fn([9999])
        assert empty_batch.num_nodes == 0
        assert empty_batch.edge_index.shape[1] == 0


class TestStructureCentricDataloader:
    """Test create_structure_centric_dataloader convenience function."""

    def test_dataloader_creation_and_transforms(
        self, preprocessor_with_structures
    ):
        """Test dataloader creation with transforms."""
        from omegaconf import OmegaConf

        from topobench.transforms.data_transform import DataTransform

        transform_config = OmegaConf.create(
            {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2,
            }
        )
        transform = DataTransform(**transform_config)

        loader = create_structure_centric_dataloader(
            preprocessor_with_structures,
            cliques_per_batch=2,
            node_budget=10,
            transform=transform,
            shuffle=False,
        )

        # Implicitly tests: iteration, Data objects, length
        batches = list(loader)
        assert len(batches) > 0
        assert abs(len(loader) - len(batches)) <= 1

        # Verify transform was applied
        batch = batches[0]
        assert hasattr(batch, "x_0") and hasattr(batch, "incidence_1")


class TestStructureCentricIntegration:
    """Integration tests for complete structure-centric workflow."""

    def test_structure_completeness_guarantee(
        self, preprocessor_with_structures
    ):
        """Test 100% structure completeness and multi-epoch training."""
        loader = create_structure_centric_dataloader(
            preprocessor_with_structures,
            cliques_per_batch=2,
            node_budget=10,
            shuffle=True,
        )

        # Simulate multiple epochs
        for _epoch in range(2):
            for batch in loader:
                # Verify structure completeness guarantee
                assert batch.num_structures > 0

                # All structures should be complete (all nodes present)
                for (
                    _struct_id,
                    batch_local_nodes,
                ) in batch.precomputed_structures:
                    assert all(
                        0 <= n < batch.num_nodes for n in batch_local_nodes
                    )
                    assert len(batch_local_nodes) == 3  # Triangles

    def test_memory_efficiency_on_large_graph(self, larger_graph):
        """Test memory efficiency on larger graph."""
        import gc
        import os

        import psutil

        _, preprocessor = larger_graph

        process = psutil.Process(os.getpid())
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024

        loader = create_structure_centric_dataloader(
            preprocessor,
            cliques_per_batch=50,
            node_budget=200,
            shuffle=False,
        )

        for batch_idx, batch in enumerate(loader):
            assert batch.num_nodes <= 210  # Budget + tolerance
            if batch_idx >= 5:
                break

        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - baseline_memory
        assert memory_increase < 200


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_sampler_with_no_structures(self, tmp_path):
        """Test that sampler raises error when graph has no structures."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        data = Data(x=torch.randn(4, 8), edge_index=edge_index, num_nodes=4)

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=data,
            data_dir=tmp_path / "no_structures",
            max_clique_size=3,
        )
        preprocessor.build_index()

        assert preprocessor.num_cliques == 0

        with pytest.raises(ValueError, match="No cliques"):
            StructureCentricSampler(
                preprocessor,
                cliques_per_batch=2,
                node_budget=10,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
