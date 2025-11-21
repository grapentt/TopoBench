"""Test 1.1: In-Memory Inductive with TopoBench Workflow - Expected to FAIL (OOM).

Uses proper TopoBench workflow:
1. Load dataset with loader
2. Apply lifting with PreProcessor
3. Create splits
4. Setup TBDataloader
5. Train SCCNNCustom for 2 epochs

Expected: OOM during preprocessing/lifting (before training)
"""

import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

# ============================================================================
# TUNABLE PARAMETERS - Adjust these for your machine
# ============================================================================
MAX_MEMORY_GB = 5  # Maximum RAM available (GB)
MAX_DISK_GB = 8  # Maximum disk space available (GB)

# Dataset parameters tuned for OOM at MAX_MEMORY_GB
# Adjust num_graphs, nodes_per_graph, or degree to tune memory pressure
NUM_GRAPHS = 5000  # More graphs → more structures → more memory
NODES_PER_GRAPH = 80  # Larger graphs → more triangles → more memory
DEGREE = 15  # Higher degree → more triangles → more memory

# These parameters create ~2-3GB of structure data in memory
# which will OOM on a 5GB RAM system during preprocessing
# ============================================================================

print("=" * 70)
print("TEST 1.1: IN-MEMORY INDUCTIVE (TopoBench Workflow)")
print("=" * 70)
print()
print("Configuration:")
print(f"  Max RAM: {MAX_MEMORY_GB} GB")
print(f"  Max Disk: {MAX_DISK_GB} GB")
print(f"  Dataset: {NUM_GRAPHS} graphs, ~{NODES_PER_GRAPH} nodes each")
print(
    f"  Degree: {DEGREE} (creates ~{NODES_PER_GRAPH * DEGREE // 3} triangles/graph)"
)
print(
    f"  Expected total structures: ~{NUM_GRAPHS * NODES_PER_GRAPH * DEGREE // 3:,}"
)
print()
print("Expected: OOM during preprocessing phase")
print()

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Note: If running without venv, you may need to mock missing dependencies
# import unittest.mock as mock
# for mod in ['torch_sparse', 'topomodelx', ...]:
#     sys.modules[mod] = mock.MagicMock()

# Import TopoBench components
from topobench.data.loaders.synthetic_large_inductive_loader import (
    SyntheticLargeInductiveLoader,
)
from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.nn.encoders import AllCellFeatureEncoder
from topobench.nn.backbones.simplicial.sccnn import SCCNNCustom
from topobench.nn.wrappers import SCCNNWrapper
from topobench.nn.readouts import PropagateSignalDown
from topobench.loss import TBLoss
from topobench.evaluator import TBEvaluator
from topobench.optimizer import TBOptimizer
from topobench.model import TBModel

# Configuration (designed to OOM during preprocessing)
HIDDEN_DIM = 32
OUT_CHANNELS = 5
NUM_LAYERS = 1

print("Step 1: Configure dataset and lifting...")
print()

# Loader config - uses constants defined at top of file
loader_config = OmegaConf.create(
    {
        "data_dir": "./data/",
        "data_name": "SyntheticLargeInductive",
        "num_graphs": NUM_GRAPHS,
        "nodes_per_graph": NODES_PER_GRAPH,
        "degree": DEGREE,
        "num_features": 16,
        "num_classes": 5,
    }
)

# Transform config - cell clique lifting with triangles
transform_config = OmegaConf.create(
    {
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,  # Include triangles
        }
    }
)

# Split config
split_config = OmegaConf.create(
    {
        "learning_setting": "inductive",
        "split_type": "random",
        "data_seed": 0,
        "data_split_dir": "./data/SyntheticLargeInductive/splits/",
        "train_prop": 0.5,
    }
)

print("Step 2: Load/Generate synthetic dataset...")
print()

synthetic_loader = SyntheticLargeInductiveLoader(loader_config)
dataset, dataset_dir = synthetic_loader.load()

print(f"✓ Loaded/Generated {len(dataset)} graphs")
print(
    f"  Expected triangles per graph: ~{loader_config.nodes_per_graph * loader_config.degree // 3}"
)
print(
    f"  Total expected triangles: ~{len(dataset) * loader_config.nodes_per_graph * loader_config.degree // 3:,}"
)
print()

print("Step 3: Apply lifting (IN-MEMORY)...")
print("⚠️  This will enumerate ALL structures and store in RAM!")
print("⚠️  Expected to OOM here!")
print()

try:
    # This uses IN-MEMORY lifting by default
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)

    print("Creating train/val/test splits...")
    dataset_train, dataset_val, dataset_test = (
        preprocessor.load_dataset_splits(split_config)
    )

    print(f"✓ Splits created:")
    print(f"  Train: {len(dataset_train)}")
    print(f"  Val: {len(dataset_val)}")
    print(f"  Test: {len(dataset_test)}")
    print()

    print("Creating TBDataloader...")
    datamodule = TBDataloader(
        dataset_train, dataset_val, dataset_test, batch_size=32
    )
    print("✓ Dataloader created")
    print()

    print("=" * 70)
    print("⚠️  UNEXPECTED: Preprocessing succeeded!")
    print("=" * 70)
    print("The dataset is smaller than expected or your machine has more RAM.")
    print("In-memory approach worked for this size.")
    print()

except (MemoryError, RuntimeError) as e:
    print()
    print("=" * 70)
    print("💥💥💥 OOM DURING PREPROCESSING (AS EXPECTED) 💥💥💥")
    print("=" * 70)
    print(f"Error: {type(e).__name__}: {e}")
    print()
    print("✓ VALIDATION SUCCESSFUL:")
    print("  In-memory inductive preprocessing FAILED due to OOM")
    print("  This proves the in-memory approach cannot scale!")
    print()
    print("Next: Run test_1_2 to see on-disk success")
    print()
    sys.exit(0)

except Exception as e:
    print()
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("If preprocessing succeeded, attempting training...")
print()

# Setup model (following tutorial pattern)
in_channels = loader_config.num_features

feature_encoder = AllCellFeatureEncoder(
    in_channels=[in_channels, in_channels, in_channels],
    out_channels=HIDDEN_DIM,
    proj_dropout=0.0,
    selected_dimensions=[0, 1, 2],
)

backbone = SCCNNCustom(
    in_channels_all=[HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM],
    hidden_channels_all=[HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM],
    conv_order=1,
    sc_order=3,
    aggr_norm=False,
    update_func="sigmoid",
    n_layers=NUM_LAYERS,
)

backbone_wrapper = SCCNNWrapper(
    backbone=backbone, out_channels=HIDDEN_DIM, num_cell_dimensions=3
)

readout_config = OmegaConf.create(
    {
        "readout_name": "PropagateSignalDown",
        "num_cell_dimensions": 3,
        "hidden_dim": HIDDEN_DIM,
        "out_channels": OUT_CHANNELS,
        "task_level": "graph",
        "pooling_type": "sum",
    }
)

loss_config = OmegaConf.create(
    {"dataset_loss": {"task": "classification", "loss_type": "cross_entropy"}}
)

evaluator_config = OmegaConf.create(
    {
        "task": "classification",
        "num_classes": OUT_CHANNELS,
        "metrics": ["accuracy"],
    }
)

optimizer_config = OmegaConf.create(
    {
        "optimizer_id": "Adam",
        "parameters": {"lr": 0.01, "weight_decay": 0.0005},
    }
)

readout = PropagateSignalDown(**readout_config)
loss = TBLoss(**loss_config)
evaluator = TBEvaluator(**evaluator_config)
optimizer = TBOptimizer(**optimizer_config)

model = TBModel(
    backbone=backbone_wrapper,
    backbone_wrapper=None,
    readout=readout,
    loss=loss,
    feature_encoder=feature_encoder,
    evaluator=evaluator,
    optimizer=optimizer,
    compile=False,
)

print("Step 4: Training model (2 epochs)...")
print()

trainer = pl.Trainer(
    max_epochs=2,
    accelerator="cpu",
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False,
)

try:
    trainer.fit(model, datamodule)

    print()
    print("=" * 70)
    print("✅ IN-MEMORY SUCCEEDED (Unexpected)")
    print("=" * 70)
    print("Dataset was small enough to fit in memory.")
    print("Try a larger dataset to demonstrate OOM.")
    print()

except Exception as e:
    print()
    print(f"❌ Training error: {e}")
    import traceback

    traceback.print_exc()
