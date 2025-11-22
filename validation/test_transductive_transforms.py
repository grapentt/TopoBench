#!/usr/bin/env python3
"""Simple standalone test for OnDiskTransductivePreprocessor transform support.

This script demonstrates that arbitrary transforms can be applied during
batch collation in transductive learning.

Usage:
    python validation/test_transductive_transforms.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add topobench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

from topobench.data.preprocessor import OnDiskTransductivePreprocessor
from topobench.dataloader import OnDiskTransductiveCollate


def create_test_graph(num_nodes=30, edge_prob=0.3):
    """Create a test graph for validation."""
    edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob)
    
    data = Data(
        x=torch.randn(num_nodes, 16),
        edge_index=edge_index,
        y=torch.randint(0, 3, (num_nodes,)),
        train_mask=torch.zeros(num_nodes, dtype=torch.bool),
        val_mask=torch.zeros(num_nodes, dtype=torch.bool),
        test_mask=torch.zeros(num_nodes, dtype=torch.bool),
    )
    
    # Set splits
    split_train = int(0.6 * num_nodes)
    split_val = int(0.8 * num_nodes)
    
    data.train_mask[:split_train] = True
    data.val_mask[split_train:split_val] = True
    data.test_mask[split_val:] = True
    
    return data


def test_without_transforms():
    """Test collate function WITHOUT transforms (baseline)."""
    print("\n" + "=" * 80)
    print("TEST 1: Collate WITHOUT Transforms (Baseline)")
    print("=" * 80)
    
    temp_dir = tempfile.mkdtemp()
    try:
        graph_data = create_test_graph(num_nodes=30)
        
        # Create preprocessor WITHOUT transforms
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=graph_data,
            data_dir=Path(temp_dir) / "no_transforms",
            transforms_config=None,
            max_structure_size=3,
        )
        
        print("Building index...")
        preprocessor.build_index()
        print(f"✓ Indexed {preprocessor.num_structures} structures")
        
        # Create collate function
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        print(f"Transform initialized: {collate_fn.transform is not None}")
        
        # Query batch
        node_ids = [0, 1, 2, 3, 4]
        print(f"\nQuerying batch with nodes: {node_ids}")
        batch = collate_fn([node_ids])
        
        print(f"✓ Batch created with {batch.num_nodes} nodes")
        print(f"  - x.shape: {batch.x.shape}")
        print(f"  - edge_index.shape: {batch.edge_index.shape}")
        print(f"  - Has x_all: {hasattr(batch, 'x_all')}")
        print(f"  - Has laplacian_all: {hasattr(batch, 'laplacian_all')}")
        print(f"  - Has incidence_all: {hasattr(batch, 'incidence_all')}")
        
        # Verify NO simplicial complex structures
        assert not hasattr(batch, 'x_all'), "Should not have x_all without transforms"
        assert not hasattr(batch, 'laplacian_all'), "Should not have laplacian_all"
        assert not hasattr(batch, 'incidence_all'), "Should not have incidence_all"
        
        print("\n✓ TEST 1 PASSED: No transforms applied as expected")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_with_transforms():
    """Test collate function WITH transforms."""
    print("\n" + "=" * 80)
    print("TEST 2: Collate WITH Transforms (SimplicialCliqueLifting)")
    print("=" * 80)
    
    temp_dir = tempfile.mkdtemp()
    try:
        graph_data = create_test_graph(num_nodes=30)
        
        # Configure transforms
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2,
            }
        })
        
        # Create preprocessor WITH transforms
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=graph_data,
            data_dir=Path(temp_dir) / "with_transforms",
            transforms_config=transforms_config,
            max_structure_size=3,
        )
        
        print("Building index...")
        preprocessor.build_index()
        print(f"✓ Indexed {preprocessor.num_structures} structures")
        
        # Create collate function
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        print(f"Transform initialized: {collate_fn.transform is not None}")
        
        # Query batch
        node_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print(f"\nQuerying batch with {len(node_ids)} nodes: {node_ids}")
        batch = collate_fn([node_ids])
        
        print(f"✓ Batch created with {batch.num_nodes} nodes")
        print(f"  - x.shape: {batch.x.shape}")
        print(f"  - edge_index.shape: {batch.edge_index.shape}")
        print(f"  - Has x_all: {hasattr(batch, 'x_all')}")
        print(f"  - Has laplacian_all: {hasattr(batch, 'laplacian_all')}")
        print(f"  - Has incidence_all: {hasattr(batch, 'incidence_all')}")
        
        # Verify simplicial complex structures exist
        # Note: Transforms create individual attributes (x_0, x_1, etc.), not grouped tuples
        assert hasattr(batch, 'x_0'), "Should have x_0 after transforms"
        assert hasattr(batch, 'x_1'), "Should have x_1 after transforms"
        assert hasattr(batch, 'incidence_1'), "Should have incidence_1 after transforms"
        
        # Check for laplacians
        assert hasattr(batch, 'hodge_laplacian_0') or hasattr(batch, 'down_laplacian_1'), \
            "Should have laplacian matrices after transforms"
        
        # Verify structure
        print(f"\nSimplicial complex structures (individual attributes):")
        print(f"  - x_0: {batch.x_0.shape if hasattr(batch, 'x_0') else 'N/A'}")
        print(f"  - x_1: {batch.x_1.shape if hasattr(batch, 'x_1') else 'N/A'}")
        if hasattr(batch, 'x_2'):
            print(f"  - x_2: {batch.x_2.shape}")
        
        print(f"\n  Laplacians:")
        if hasattr(batch, 'hodge_laplacian_0'):
            L0 = batch.hodge_laplacian_0
            print(f"  - hodge_laplacian_0: {L0.shape if hasattr(L0, 'shape') else 'sparse'}")
        if hasattr(batch, 'down_laplacian_1'):
            L1_down = batch.down_laplacian_1
            print(f"  - down_laplacian_1: {L1_down.shape if hasattr(L1_down, 'shape') else 'sparse'}")
        if hasattr(batch, 'up_laplacian_1'):
            L1_up = batch.up_laplacian_1
            print(f"  - up_laplacian_1: {L1_up.shape if hasattr(L1_up, 'shape') else 'sparse'}")
        
        print(f"\n  Incidences:")
        if hasattr(batch, 'incidence_1'):
            B1 = batch.incidence_1
            print(f"  - incidence_1: {B1.shape if hasattr(B1, 'shape') else 'sparse'}")
        if hasattr(batch, 'incidence_2'):
            B2 = batch.incidence_2
            print(f"  - incidence_2: {B2.shape if hasattr(B2, 'shape') else 'sparse'}")
        
        # Verify features are preserved
        original_features = batch.x_0
        expected_features = graph_data.x[node_ids]
        assert torch.allclose(original_features, expected_features), \
            "Node features should be preserved"
        print(f"\n✓ Node features preserved correctly")
        
        print("\n✓ TEST 2 PASSED: Transforms applied successfully!")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_multiple_batches():
    """Test that transforms work across multiple batches."""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Batches with Transforms")
    print("=" * 80)
    
    temp_dir = tempfile.mkdtemp()
    try:
        graph_data = create_test_graph(num_nodes=50)
        
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2,
            }
        })
        
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=graph_data,
            data_dir=Path(temp_dir) / "multi_batch",
            transforms_config=transforms_config,
            max_structure_size=3,
        )
        
        print("Building index...")
        preprocessor.build_index()
        print(f"✓ Indexed {preprocessor.num_structures} structures")
        
        collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset=preprocessor,
            fully_contained=True,
        )
        
        # Test multiple batches
        batches = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 15, 20, 25, 30],
            [0, 10, 20, 30, 40],
        ]
        
        print(f"\nTesting {len(batches)} different batches...")
        for i, node_ids in enumerate(batches, 1):
            batch = collate_fn([node_ids])
            
            assert hasattr(batch, 'x_0'), f"Batch {i} missing x_0"
            assert hasattr(batch, 'x_1'), f"Batch {i} missing x_1"
            assert hasattr(batch, 'incidence_1'), f"Batch {i} missing incidence_1"
            
            print(f"  ✓ Batch {i}: {len(node_ids)} nodes → simplicial complex structures created")
        
        print("\n✓ TEST 3 PASSED: All batches processed correctly!")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING: OnDiskTransductivePreprocessor Transform Support")
    print("=" * 80)
    print("\nThis test validates that arbitrary transforms can be applied")
    print("during batch collation in transductive learning.")
    
    try:
        # Run tests
        test1_passed = test_without_transforms()
        test2_passed = test_with_transforms()
        test3_passed = test_multiple_batches()
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Test 1 (No transforms): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
        print(f"Test 2 (With transforms): {'✓ PASSED' if test2_passed else '✗ FAILED'}")
        print(f"Test 3 (Multiple batches): {'✓ PASSED' if test3_passed else '✗ FAILED'}")
        
        if all([test1_passed, test2_passed, test3_passed]):
            print("\n" + "=" * 80)
            print("✓ ALL TESTS PASSED!")
            print("=" * 80)
            print("\nConclusion: OnDiskTransductivePreprocessor now supports")
            print("arbitrary transforms applied at batch-time during collation.")
            print("This enables full TopoBench pipeline for transductive learning!")
            return 0
        else:
            print("\n✗ SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
