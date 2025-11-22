"""Training script for OGBN-products using on-disk transductive learning with TopoBench.

This script demonstrates large-scale transductive learning on OGBN-products
(2.4M nodes, 61M edges) using:
- On-disk structure indexing and mini-batch training
- TopoBench SCCNNCustom model for simplicial complex learning
- PyTorch Lightning for training
- Constant memory usage via on-disk preprocessing

Key features:
- Integrates TopoBench's on-disk preprocessor with Lightning
- Mini-batch training with on-demand structure querying
- Scales to graphs much larger than RAM
- Uses proper TopoBench model architecture

This script demonstrates proper TopoBench framework patterns:
1. Uses TBModel (not custom Lightning modules)
2. Uses TBLoss, TBOptimizer (TopoBench's standard components)
3. Uses MiniBatchTransductiveDataset wrapper (follows TopoBench dataset pattern)
4. Integrates with on-disk preprocessing for memory efficiency

NOTE: Full simplicial complex support requires:
- Proper lifting transforms applied during preprocessing
- Collate function that creates simplicial complex batches (x_all, laplacian_all, incidence_all)

For full TopoBench pipeline with Hydra configs, see: topobench/run.py

Usage:
    python examples/train_ogbn_products_ondisk.py --max_epochs 10 --batch_size 1024
"""

import argparse
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from topobench.data.loaders import OGBNProductsLoader
from topobench.data.preprocessor import OnDiskTransductivePreprocessor
from topobench.dataloader import NodeBatchSampler, OnDiskTransductiveCollate
from topobench.loss.loss import TBLoss
from topobench.model.model import TBModel
from topobench.nn.backbones.simplicial import SCCNNCustom
from topobench.nn.readouts.simplicial_readout import SimplicialReadout
from topobench.optimizer.optimizer import TBOptimizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train on OGBN-products with on-disk transductive learning"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/ogbn_products",
        help="Directory for dataset storage",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="./data/ogbn_products_index",
        help="Directory for structure index",
    )
    parser.add_argument(
        "--max_structure_size",
        type=int,
        default=3,
        help="Maximum structure size to index (3=triangles)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Mini-batch size"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GNN layers"
    )
    parser.add_argument(
        "--force_rebuild_index",
        action="store_true",
        help="Force rebuild of structure index",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    return parser.parse_args()


def create_ogbn_model(in_channels, hidden_channels, out_channels, num_layers=2, lr=0.01, weight_decay=0.0):
    """Create TopoBench model for OGBN-products.
    
    Uses proper TopoBench TBModel with separate components:
    - SCCNNCustom backbone for simplicial complex learning
    - SimplicialReadout for node classification
    - TBLoss with cross entropy
    - TBOptimizer with Adam
    """
    # SCCNN backbone
    in_channels_all = (in_channels, hidden_channels, hidden_channels)
    hidden_channels_all = (hidden_channels, hidden_channels, hidden_channels)
    
    backbone = SCCNNCustom(
        in_channels_all=in_channels_all,
        hidden_channels_all=hidden_channels_all,
        conv_order=1,
        sc_order=2,  # Triangles
        n_layers=num_layers,
    )
    
    # Readout for node classification
    readout = SimplicialReadout(
        in_channels=hidden_channels,
        out_channels=out_channels,
        task_level="node",
    )
    
    # Loss function
    loss = TBLoss(
        dataset_loss={
            "task": "classification",
            "loss_type": "cross_entropy",
        }
    )
    
    # Optimizer
    optimizer = TBOptimizer(
        optimizer_id="Adam",
        parameters={"lr": lr, "weight_decay": weight_decay},
    )
    
    # Create TBModel (TopoBench's standard Lightning module)
    model = TBModel(
        backbone=backbone,
        readout=readout,
        loss=loss,
        optimizer=optimizer,
    )
    
    return model


class MiniBatchTransductiveDataset(IterableDataset):
    """Dataset wrapper for mini-batch transductive learning.
    
    Wraps NodeBatchSampler and OnDiskTransductiveCollate to work with
    standard PyTorch DataLoader and TopoBench's training pipeline.
    This follows TopoBench's pattern of using dataset wrappers rather than
    custom Lightning modules.
    """
    
    def __init__(self, ondisk_dataset, graph_data, batch_size, shuffle=True, mask=None):
        self.ondisk_dataset = ondisk_dataset
        self.graph_data = graph_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mask = mask
        
        self.sampler = NodeBatchSampler(
            num_nodes=graph_data.num_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            mask=mask,
        )
        
        self.collate_fn = OnDiskTransductiveCollate(
            ondisk_dataset, fully_contained=True
        )
    
    def __iter__(self):
        for node_batch in self.sampler:
            yield self.collate_fn([node_batch])
    
    def __len__(self):
        return len(self.sampler)


def main():
    """Main training script."""
    args = parse_args()

    print("=" * 80)
    print("OGBN-products On-Disk Transductive Learning")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Max structure size: {args.max_structure_size}")
    print()

    # Step 1: Load dataset
    print("[1/5] Loading OGBN-products dataset...")
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {"data_dir": args.data_dir, "data_name": "ogbn-products"}
    )

    loader = OGBNProductsLoader(config)
    dataset, data_dir = loader.load()
    graph_data = dataset[0]

    print(f"✓ Loaded graph:")
    print(f"  - Nodes: {graph_data.num_nodes:,}")
    print(f"  - Edges: {graph_data.edge_index.shape[1]:,}")
    print(f"  - Features: {graph_data.x.shape[1]}")
    print(f"  - Classes: {graph_data.y.max().item() + 1}")
    print(f"  - Train nodes: {graph_data.train_mask.sum():,}")
    print(f"  - Val nodes: {graph_data.val_mask.sum():,}")
    print(f"  - Test nodes: {graph_data.test_mask.sum():,}")
    print()

    # Step 2: Create on-disk transductive dataset with transforms
    print("[2/5] Creating on-disk transductive dataset...")
    
    # Configure lifting transforms for simplicial complex
    transforms_config = OmegaConf.create({
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,
        }
    })
    
    ondisk_dataset = OnDiskTransductivePreprocessor(
        graph_data=graph_data,
        data_dir=args.index_dir,
        transforms_config=transforms_config,
        max_structure_size=args.max_structure_size,
        force_rebuild=args.force_rebuild_index,
    )

    # Build index (one-time operation, cached on disk)
    print("Building structure index (this may take a few minutes on first run)...")
    ondisk_dataset.build_index()
    print(
        f"✓ Index built: {ondisk_dataset.num_structures:,} structures indexed"
    )
    print()

    # Step 3: Create datasets for train/val/test
    print("[3/5] Creating mini-batch datasets...")
    train_dataset = MiniBatchTransductiveDataset(
        ondisk_dataset=ondisk_dataset,
        graph_data=graph_data,
        batch_size=args.batch_size,
        shuffle=True,
        mask=graph_data.train_mask,
    )
    
    val_dataset = MiniBatchTransductiveDataset(
        ondisk_dataset=ondisk_dataset,
        graph_data=graph_data,
        batch_size=args.batch_size,
        shuffle=False,
        mask=graph_data.val_mask,
    )
    
    test_dataset = MiniBatchTransductiveDataset(
        ondisk_dataset=ondisk_dataset,
        graph_data=graph_data,
        batch_size=args.batch_size,
        shuffle=False,
        mask=graph_data.test_mask,
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)
    test_loader = DataLoader(test_dataset, batch_size=None)
    
    print(f"✓ Datasets created (batch size: {args.batch_size})")
    print()

    # Step 4: Create model using TopoBench TBModel
    print("[4/5] Creating SCCNNCustom model with TopoBench TBModel...")
    model = create_ogbn_model(
        in_channels=graph_data.x.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=graph_data.y.max().item() + 1,
        num_layers=args.num_layers,
        lr=args.lr,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    print(f"  - Using TopoBench TBModel with:")
    print(f"    * Backbone: SCCNNCustom (Simplicial Complex CNN)")
    print(f"    * Readout: SimplicialReadout (node-level)")
    print(f"    * Loss: TBLoss (cross_entropy)")
    print(f"    * Optimizer: TBOptimizer (Adam)")
    print()

    # Step 5: Training with Lightning using TopoBench pattern
    print("[5/5] Training with PyTorch Lightning...")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="ogbn-products-best",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=5,
        mode="min",
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\nEvaluating on test set...")
    test_results = trainer.test(model, test_loader)
    print(f"✓ Test results: {test_results}")

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(
        "\nKey achievement: Trained on 2.4M node graph with constant memory usage!"
    )
    print(
        "In-memory approach would require ~10-30GB just for structures."
    )


if __name__ == "__main__":
    main()
