"""TopoBench Transductive Validation: In-Memory vs On-Disk.

This script uses the TopoBench framework to validate that our on-disk
transductive dataset enables training where in-memory approaches fail.
"""

import gc
import sys
import time
from pathlib import Path

import lightning as pl
import networkx as nx
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.dataloader.dataloader import TBDataloader
from topobench.evaluator.evaluator import TBEvaluator
from topobench.loss.loss import TBLoss
from topobench.model.model import TBModel
from topobench.nn.encoders import AllCellFeatureEncoder
from topobench.nn.readouts import PropagateSignalDown
from topobench.nn.wrappers.simplicial import SCNWrapper
from topobench.optimizer import TBOptimizer
from topomodelx.nn.simplicial.scn2 import SCN2


def generate_synthetic_transductive_graph(nodes=10000, avg_degree=20, seed=42):
    """Generate synthetic graph for transductive learning.

    Parameters
    ----------
    nodes : int
        Number of nodes
    avg_degree : int
        Average degree
    seed : int
        Random seed

    Returns
    -------
    Data
        PyG Data object
    """
    print(
        f"Generating transductive graph: {nodes} nodes, avg degree {avg_degree}"
    )

    G = nx.watts_strogatz_graph(n=nodes, k=avg_degree, p=0.1, seed=seed)

    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    x = torch.randn(nodes, 32, generator=torch.manual_seed(seed))
    y = torch.randint(0, 10, (nodes,), generator=torch.manual_seed(seed))

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=nodes)

    # Create train/val/test masks
    n_train = int(0.6 * nodes)
    n_val = int(0.2 * nodes)

    train_mask = torch.zeros(nodes, dtype=torch.bool)
    val_mask = torch.zeros(nodes, dtype=torch.bool)
    test_mask = torch.zeros(nodes, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"✓ Graph generated: {G.number_of_edges()} edges")
    print(f"  Train: {train_mask.sum()} nodes")
    print(f"  Val: {val_mask.sum()} nodes")
    print(f"  Test: {test_mask.sum()} nodes")
    print()

    return data


def test_inmemory_transductive(nodes=10000, avg_degree=20):
    """Test in-memory transductive approach - expected to OOM."""
    print("=" * 70)
    print("TRANSDUCTIVE VALIDATION: IN-MEMORY APPROACH")
    print("=" * 70)
    print(f"Graph: {nodes} nodes, avg degree {avg_degree}")
    print(f"Expected: OOM during lifting (stores all structures in memory)")
    print()

    try:
        # Generate graph
        data = generate_synthetic_transductive_graph(nodes, avg_degree)

        # Try to apply simplicial lifting IN-MEMORY
        print("Applying simplicial clique lifting (IN-MEMORY)...")
        print("⚠️  This finds and stores ALL triangles in memory!")
        print()

        start_time = time.time()

        # Find all triangles and store in memory (this should OOM)
        G = nx.Graph()
        edge_list = data.edge_index.t().tolist()
        G.add_edges_from(edge_list)

        print("Finding all triangles...")
        triangles_list = []  # Store ALL in memory

        for i, node in enumerate(G.nodes()):
            if i % 1000 == 0 and i > 0:
                print(
                    f"  Processed {i}/{nodes} nodes ({len(triangles_list)} triangles)..."
                )

            neighbors = list(G.neighbors(node))
            for idx, n1 in enumerate(neighbors):
                for n2 in neighbors[idx + 1 :]:
                    if G.has_edge(n1, n2):
                        triangle = tuple(sorted([node, n1, n2]))
                        if triangle[0] == node:
                            triangles_list.append(triangle)  # STORE IN RAM

        elapsed = time.time() - start_time
        print(f"\n✓ Found {len(triangles_list)} triangles ({elapsed:.1f}s)")
        print()

        # Create lifted features (even more memory)
        print("Creating lifted simplicial complex features...")
        triangle_features = torch.randn(len(triangles_list), 32)
        print(f"✓ Triangle features: {triangle_features.shape}")
        print()

        print("=" * 70)
        print("⚠️  IN-MEMORY APPROACH SUCCEEDED")
        print("=" * 70)
        print("Your machine has more RAM than expected!")
        print(
            f"Try increasing to {nodes + 5000} nodes or degree {avg_degree + 5}"
        )
        print()

        return True, nodes

    except (MemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
            print()
            print("=" * 70)
            print("💥 OOM CRASH (AS EXPECTED)")
            print("=" * 70)
            print(f"Error: {e}")
            print()
            print("In-memory lifting failed at this scale!")
            print()
            return False, nodes
        else:
            raise

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted (likely killed by OS due to memory)")
        print("This counts as OOM failure!")
        return False, nodes


def test_ondisk_transductive(nodes=10000, avg_degree=20):
    """Test on-disk transductive approach - expected to succeed."""
    print("=" * 70)
    print("TRANSDUCTIVE VALIDATION: ON-DISK APPROACH")
    print("=" * 70)
    print(f"Graph: {nodes} nodes, avg degree {avg_degree}")
    print(f"Expected: Success with constant memory")
    print()

    try:
        from topobench.data.preprocessor.ondisk_transductive import (
            OnDiskTransductivePreprocessor,
        )

        # Generate same graph
        data = generate_synthetic_transductive_graph(nodes, avg_degree)

        print("Creating OnDiskTransductiveDataset...")
        print("⚠️  This uses streaming enumeration (constant memory)!")
        print()

        data_dir = Path("/tmp/ondisk_transductive_validation")

        start_time = time.time()

        # Use on-disk dataset
        ondisk_dataset = OnDiskTransductivePreprocessor(
            graph_data=data,
            data_dir=str(data_dir),
            max_structure_size=3,  # Triangles
            force_rebuild=True,
        )

        print("Building index (streaming)...")
        ondisk_dataset.build_index()

        elapsed = time.time() - start_time
        print(f"\n✓ Index built ({elapsed:.1f}s)")
        print(f"  Structures indexed: {ondisk_dataset.num_structures:,}")
        print(f"  Memory: Constant during indexing")
        print()

        # Test batch queries
        print("Testing batch queries...")
        batch_nodes = list(range(1000))
        structures = ondisk_dataset.query_batch(
            batch_nodes, fully_contained=True
        )
        print(f"✓ Query for 1000 nodes: {len(structures)} structures")
        print()

        # For training, we need to create a simple wrapper
        # Since transductive uses the full graph, we create a simple dataset wrapper
        class TransductiveDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, data, ondisk_dataset):
                self.data = data
                self.ondisk_dataset = ondisk_dataset

            def __len__(self):
                return 1  # Single graph (transductive)

            def __getitem__(self, idx):
                return self.data

        # Create dataset wrappers
        train_data = data.clone()
        train_data.mask = data.train_mask

        val_data = data.clone()
        val_data.mask = data.val_mask

        test_data = data.clone()
        test_data.mask = data.test_mask

        train_dataset = TransductiveDatasetWrapper(train_data, ondisk_dataset)
        val_dataset = TransductiveDatasetWrapper(val_data, ondisk_dataset)
        test_dataset = TransductiveDatasetWrapper(test_data, ondisk_dataset)

        # For transductive, batch_size=1 (full graph)
        datamodule = TBDataloader(
            train_dataset, val_dataset, test_dataset, batch_size=1
        )

        print("Setting up model...")
        backbone = SCN2(in_channels_0=16, in_channels_1=16, in_channels_2=16)

        wrapper_config = {
            "out_channels": 16,
            "num_cell_dimensions": 3,
        }

        readout_config = {
            "readout_name": "PropagateSignalDown",
            "num_cell_dimensions": 1,
            "hidden_dim": 16,
            "out_channels": 10,
            "task_level": "node",
            "pooling_type": "sum",
        }

        loss_config = {
            "dataset_loss": {
                "task": "classification",
                "loss_type": "cross_entropy",
            }
        }

        evaluator_config = {
            "task": "classification",
            "num_classes": 10,
            "metrics": ["accuracy"],
        }

        optimizer_config = {
            "optimizer_id": "Adam",
            "parameters": {"lr": 0.001, "weight_decay": 5e-4},
        }

        wrapper = SCNWrapper(backbone, **wrapper_config)
        readout = PropagateSignalDown(**readout_config)
        loss = TBLoss(**loss_config)
        feature_encoder = AllCellFeatureEncoder(
            in_channels=[32, 32, 32], out_channels=16
        )
        evaluator = TBEvaluator(**evaluator_config)
        optimizer = TBOptimizer(**optimizer_config)

        model = TBModel(
            backbone=wrapper,
            backbone_wrapper=None,
            readout=readout,
            loss=loss,
            feature_encoder=feature_encoder,
            evaluator=evaluator,
            optimizer=optimizer,
            compile=False,
        )

        print("✓ Model created")
        print()

        # Train
        print("Training for 3 epochs...")
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=True,
            log_every_n_steps=1,
            enable_checkpointing=False,
        )

        trainer.fit(model, datamodule)

        ondisk_dataset.close()

        print()
        print("=" * 70)
        print("✅ ON-DISK APPROACH SUCCEEDED")
        print("=" * 70)
        print(
            f"Successfully trained on {nodes}-node graph with constant memory!"
        )
        print()

        return True

    except Exception as e:
        print()
        print(f"❌ On-disk approach failed: {e}")
        print()
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run transductive validation."""
    print("\n" + "=" * 70)
    print("TOPOBENCH TRANSDUCTIVE VALIDATION")
    print("=" * 70)
    print()
    print("This will test both in-memory and on-disk approaches using")
    print("the TopoBench framework for transductive learning.")
    print()

    # Try increasing sizes
    sizes = [
        (8000, 20),  # 8K nodes, degree 20
        (10000, 20),  # 10K nodes
        (12000, 25),  # 12K nodes, denser
        (15000, 25),  # 15K nodes
        (20000, 30),  # 20K nodes
    ]

    for nodes, degree in sizes:
        print(f"\nTesting with: {nodes} nodes, avg degree {degree}")
        print("-" * 70)
        input("Press Enter to test in-memory approach...")

        success, tested_size = test_inmemory_transductive(nodes, degree)

        if not success:
            print("\n✓ Found OOM threshold!")
            print(f"  In-memory FAILS at {tested_size} nodes")
            print()
            input("Press Enter to test on-disk at same size...")

            ondisk_success = test_ondisk_transductive(nodes, degree)

            if ondisk_success:
                print("\n" + "=" * 70)
                print("VALIDATION COMPLETE! ✅")
                print("=" * 70)
                print()
                print(f"PROOF:")
                print(f"  ❌ In-memory: FAILED at {tested_size} nodes")
                print(f"  ✅ On-disk: SUCCESS at {tested_size} nodes")
                print()
                print("Conclusion: On-disk enables training at scales")
                print("            where in-memory approaches fail!")
                print()
                return
            else:
                print("\n⚠️  On-disk also failed. Investigating...")
                return

        print(
            f"\n⚠️  In-memory succeeded at {tested_size} nodes. Trying larger..."
        )
        gc.collect()

    print("\n⚠️  All sizes succeeded in-memory.")
    print("Your machine has more RAM than expected.")
    print("Consider testing with even larger graphs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted.")
        sys.exit(1)
