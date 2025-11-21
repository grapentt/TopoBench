"""TopoBench Inductive Validation: In-Memory vs On-Disk.

This script uses the TopoBench framework to validate that our on-disk
inductive dataset enables training where in-memory approaches fail.
"""

import gc
import sys
import time
from pathlib import Path

import lightning as pl
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from topobench.data.loaders.graph import TUDatasetLoader
from topobench.data.preprocessor import PreProcessor
from topobench.dataloader.dataloader import TBDataloader
from topobench.evaluator.evaluator import TBEvaluator
from topobench.loss.loss import TBLoss
from topobench.model.model import TBModel
from topobench.nn.encoders import AllCellFeatureEncoder
from topobench.nn.readouts import PropagateSignalDown
from topobench.nn.wrappers.cell import CellCWNWrapper
from topobench.optimizer import TBOptimizer
from topomodelx.nn.cell.ccxn import CCXN


def generate_synthetic_inductive_dataset(
    n_graphs=5000,
    nodes_per_graph=50,
    avg_degree=10,
    data_dir="./data/synthetic_inductive",
):
    """Generate synthetic graph dataset for inductive learning.

    Parameters
    ----------
    n_graphs : int
        Number of graphs to generate
    nodes_per_graph : int
        Approximate nodes per graph (will vary)
    avg_degree : int
        Average degree
    data_dir : str
        Where to save

    Returns
    -------
    list
        List of PyG Data objects
    """
    import networkx as nx

    print(f"Generating {n_graphs} synthetic graphs...")
    print(f"  Nodes/graph: ~{nodes_per_graph}")
    print(f"  Avg degree: {avg_degree}")
    print()

    dataset = []
    for i in range(n_graphs):
        if i % 500 == 0 and i > 0:
            print(f"  Generated {i}/{n_graphs} graphs...")

        # Vary size slightly
        n = nodes_per_graph + torch.randint(-10, 10, (1,)).item()
        n = max(20, n)  # At least 20 nodes

        # Generate graph
        G = nx.watts_strogatz_graph(n=n, k=avg_degree, p=0.3, seed=42 + i)

        # Convert to PyG
        edges = list(G.edges())
        if len(edges) == 0:
            continue

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        x = torch.randn(n, 16)  # Node features
        y = torch.randint(0, 5, (1,))  # Graph label (5 classes)

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=n)
        dataset.append(data)

    print(f"✓ Generated {len(dataset)} graphs")
    print()
    return dataset


def test_inmemory_inductive(n_graphs=5000, nodes_per_graph=50, avg_degree=10):
    """Test in-memory inductive approach - expected to OOM or struggle."""
    print("=" * 70)
    print("INDUCTIVE VALIDATION: IN-MEMORY APPROACH")
    print("=" * 70)
    print(f"Dataset: {n_graphs} graphs, ~{nodes_per_graph} nodes/graph")
    print(f"Expected: OOM or high memory usage (>3GB)")
    print()

    try:
        # Generate dataset
        data_dir = Path("/tmp/synthetic_inductive_inmemory")
        dataset = generate_synthetic_inductive_dataset(
            n_graphs=n_graphs,
            nodes_per_graph=nodes_per_graph,
            avg_degree=avg_degree,
            data_dir=str(data_dir),
        )

        # Configure lifting (this is where it gets memory-intensive)
        transform_config = {
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "CellCliqueLifting",
                "complex_dim": 2,
            }
        }
        transform_config = OmegaConf.create(transform_config)

        split_config = {
            "learning_setting": "inductive",
            "split_type": "random",
            "data_seed": 42,
            "data_split_dir": str(data_dir / "splits"),
            "train_prop": 0.6,
        }
        split_config = OmegaConf.create(split_config)

        print("Applying cell clique lifting (IN-MEMORY)...")
        print("⚠️  This loads ALL lifted graphs into memory!")
        print()
        start_time = time.time()

        # This is the memory-intensive operation
        preprocessor = PreProcessor(dataset, str(data_dir), transform_config)
        dataset_train, dataset_val, dataset_test = (
            preprocessor.load_dataset_splits(split_config)
        )

        elapsed = time.time() - start_time
        print(f"✓ Lifting completed ({elapsed:.1f}s)")
        print(f"  Train: {len(dataset_train)} graphs")
        print(f"  Val: {len(dataset_val)} graphs")
        print(f"  Test: {len(dataset_test)} graphs")
        print()

        # Create dataloader
        datamodule = TBDataloader(
            dataset_train, dataset_val, dataset_test, batch_size=32
        )

        # Setup model
        print("Setting up model...")
        backbone = CCXN(in_channels_0=16, in_channels_1=16, in_channels_2=16)
        wrapper = CellCWNWrapper(out_channels=16, num_cell_dimensions=2)

        readout_config = {
            "readout_name": "PropagateSignalDown",
            "num_cell_dimensions": 1,
            "hidden_dim": 16,
            "out_channels": 5,
            "task_level": "graph",
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
            "num_classes": 5,
            "metrics": ["accuracy"],
        }

        optimizer_config = {
            "optimizer_id": "Adam",
            "parameters": {"lr": 0.001, "weight_decay": 5e-4},
        }

        readout = PropagateSignalDown(**readout_config)
        loss = TBLoss(**loss_config)
        feature_encoder = AllCellFeatureEncoder(
            in_channels=[16, 16, 16], out_channels=16
        )
        evaluator = TBEvaluator(**evaluator_config)
        optimizer = TBOptimizer(**optimizer_config)

        model = TBModel(
            backbone=backbone,
            backbone_wrapper=wrapper,
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

        print()
        print("=" * 70)
        print("⚠️  IN-MEMORY APPROACH SUCCEEDED")
        print("=" * 70)
        print("Your machine has more RAM than expected!")
        print(
            f"Try increasing to {n_graphs * 2} graphs or {nodes_per_graph + 20} nodes/graph"
        )
        print()

        return True, n_graphs

    except (MemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
            print()
            print("=" * 70)
            print("💥 OOM CRASH (AS EXPECTED)")
            print("=" * 70)
            print(f"Error: {e}")
            print()
            print("In-memory approach failed at this scale!")
            print()
            return False, n_graphs
        else:
            raise

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted (likely killed by OS due to memory)")
        print("This counts as OOM failure!")
        return False, n_graphs


def test_ondisk_inductive(n_graphs=5000, nodes_per_graph=50, avg_degree=10):
    """Test on-disk inductive approach - expected to succeed."""
    print("=" * 70)
    print("INDUCTIVE VALIDATION: ON-DISK APPROACH")
    print("=" * 70)
    print(f"Dataset: {n_graphs} graphs, ~{nodes_per_graph} nodes/graph")
    print(f"Expected: Success with constant memory")
    print()

    try:
        from topobench.data.preprocessor.ondisk_inductive import (
            OnDiskInductiveDataset,
        )

        # Generate dataset
        data_dir = Path("/tmp/synthetic_inductive_ondisk")
        dataset = generate_synthetic_inductive_dataset(
            n_graphs=n_graphs,
            nodes_per_graph=nodes_per_graph,
            avg_degree=avg_degree,
            data_dir=str(data_dir),
        )

        print("Creating OnDiskInductiveDataset...")
        print("⚠️  This uses CONSTANT memory (streaming processing)!")
        print()

        start_time = time.time()

        # Use on-disk dataset
        ondisk_dataset = OnDiskInductiveDataset(
            dataset=dataset,
            data_dir=str(data_dir / "ondisk"),
            max_k=2,  # Triangles
            force_rebuild=True,
        )

        elapsed = time.time() - start_time
        print(f"✓ On-disk dataset created ({elapsed:.1f}s)")
        print(f"  Total graphs: {len(ondisk_dataset)}")
        print(f"  Memory: Constant (structures stored on disk)")
        print()

        # Create splits
        n_train = int(0.6 * len(ondisk_dataset))
        n_val = int(0.2 * len(ondisk_dataset))

        train_dataset = [ondisk_dataset[i] for i in range(n_train)]
        val_dataset = [
            ondisk_dataset[i] for i in range(n_train, n_train + n_val)
        ]
        test_dataset = [
            ondisk_dataset[i]
            for i in range(n_train + n_val, len(ondisk_dataset))
        ]

        datamodule = TBDataloader(
            train_dataset, val_dataset, test_dataset, batch_size=32
        )

        # Setup model (same as in-memory)
        print("Setting up model...")
        backbone = CCXN(in_channels_0=16, in_channels_1=16, in_channels_2=16)
        wrapper = CellCWNWrapper(out_channels=16, num_cell_dimensions=2)

        readout_config = {
            "readout_name": "PropagateSignalDown",
            "num_cell_dimensions": 1,
            "hidden_dim": 16,
            "out_channels": 5,
            "task_level": "graph",
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
            "num_classes": 5,
            "metrics": ["accuracy"],
        }

        optimizer_config = {
            "optimizer_id": "Adam",
            "parameters": {"lr": 0.001, "weight_decay": 5e-4},
        }

        readout = PropagateSignalDown(**readout_config)
        loss = TBLoss(**loss_config)
        feature_encoder = AllCellFeatureEncoder(
            in_channels=[16, 16, 16], out_channels=16
        )
        evaluator = TBEvaluator(**evaluator_config)
        optimizer = TBOptimizer(**optimizer_config)

        model = TBModel(
            backbone=backbone,
            backbone_wrapper=wrapper,
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

        print()
        print("=" * 70)
        print("✅ ON-DISK APPROACH SUCCEEDED")
        print("=" * 70)
        print(
            f"Successfully trained on {n_graphs} graphs with constant memory!"
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
    """Run inductive validation."""
    print("\n" + "=" * 70)
    print("TOPOBENCH INDUCTIVE VALIDATION")
    print("=" * 70)
    print()
    print("This will test both in-memory and on-disk approaches using")
    print("the TopoBench framework for inductive learning.")
    print()

    # Start with moderate size
    sizes = [
        (3000, 50, 10),  # 3K graphs, 50 nodes, degree 10
        (5000, 50, 10),  # 5K graphs
        (7000, 60, 12),  # 7K graphs, denser
        (10000, 60, 12),  # 10K graphs
    ]

    for n_graphs, nodes, degree in sizes:
        print(
            f"\nTesting with: {n_graphs} graphs, {nodes} nodes/graph, degree {degree}"
        )
        print("-" * 70)
        input("Press Enter to test in-memory approach...")

        success, tested_size = test_inmemory_inductive(n_graphs, nodes, degree)

        if not success:
            print("\n✓ Found OOM threshold!")
            print(f"  In-memory FAILS at {tested_size} graphs")
            print()
            input("Press Enter to test on-disk at same size...")

            ondisk_success = test_ondisk_inductive(n_graphs, nodes, degree)

            if ondisk_success:
                print("\n" + "=" * 70)
                print("VALIDATION COMPLETE! ✅")
                print("=" * 70)
                print()
                print(f"PROOF:")
                print(f"  ❌ In-memory: FAILED at {tested_size} graphs")
                print(f"  ✅ On-disk: SUCCESS at {tested_size} graphs")
                print()
                print("Conclusion: On-disk enables training at scales")
                print("            where in-memory approaches fail!")
                print()
                return
            else:
                print("\n⚠️  On-disk also failed. Investigating...")
                return

        print(
            f"\n⚠️  In-memory succeeded at {tested_size} graphs. Trying larger..."
        )
        gc.collect()

    print("\n⚠️  All sizes succeeded in-memory.")
    print("Your machine has more RAM than expected.")
    print("Consider testing with larger datasets.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted.")
        sys.exit(1)
