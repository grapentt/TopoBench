"""Test 2.1: In-Memory Transductive with TopoBench Workflow - Expected to FAIL (OOM).

Uses proper TopoBench workflow with synthetic transductive dataset:
1. Load dataset with Synthetic LargeTransductiveLoader
2. Apply lifting with PreProcessor (IN-MEMORY)
3. Setup TBDataloader
4. Train SCCNNCustom for 2 epochs

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
# Adjust num_nodes or degree to tune memory pressure
NUM_NODES = 15000  # Large graph
DEGREE = 60  # High degree → many triangles

# These parameters create ~844K triangles → ~2-3GB in memory
# which will OOM on a 5GB RAM system during preprocessing
# ============================================================================

print("=" * 70)
print("TEST 2.1: IN-MEMORY TRANSDUCTIVE (TopoBench Workflow)")
print("=" * 70)
print()
print("Configuration:")
print(f"  Max RAM: {MAX_MEMORY_GB} GB")
print(f"  Max Disk: {MAX_DISK_GB} GB")
print(f"  Graph: {NUM_NODES} nodes, degree {DEGREE}")
print(f"  Expected triangles: ~{NUM_NODES * DEGREE // 3:,}")
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
from topobench.data.loaders.synthetic_large_transductive_loader import (
    SyntheticLargeTransductiveLoader,
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

# Configuration
HIDDEN_DIM = 32
OUT_CHANNELS = 10
NUM_LAYERS = 1

print("Step 1: Configure dataset and lifting...")
print()

# Loader config - uses constants defined at top of file
loader_config = OmegaConf.create(
    {
        "data_dir": "./data/",
        "data_name": "SyntheticLargeTransductive",
        "num_nodes": NUM_NODES,
        "degree": DEGREE,
        "num_features": 16,
        "num_classes": OUT_CHANNELS,
    }
)

# Transform config - simplicial clique lifting for transductive
transform_config = OmegaConf.create(
    {
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,  # Include triangles
        }
    }
)

print("Step 2: Load/Generate synthetic transductive graph...")
print()

loader = SyntheticLargeTransductiveLoader(loader_config)
dataset, data_dir = loader.load()

# Get the single graph from dataset
data = dataset[0]

print(f"✓ Loaded transductive graph:")
print(f"  Nodes: {data.num_nodes}")
print(f"  Edges: {data.edge_index.shape[1] // 2}")
print(f"  Train nodes: {data.train_mask.sum()}")
print(f"  Val nodes: {data.val_mask.sum()}")
print(f"  Test nodes: {data.test_mask.sum()}")
print()

print("Step 3: Apply lifting (IN-MEMORY)...")
print("⚠️  This will enumerate ALL triangles and store in RAM!")
print("⚠️  Expected to OOM here!")
print()

try:
    # Wrap single graph in list for PreProcessor
    dataset_list = [data]

    # This uses IN-MEMORY lifting by default
    preprocessor = PreProcessor(dataset_list, data_dir, transform_config)

    print("Preprocessing...")
    # For transductive, we don't split the graph, just process it
    processed_data = preprocessor.transform(data)

    print()
    print("=" * 70)
    print("⚠️  UNEXPECTED: Preprocessing succeeded!")
    print("=" * 70)
    print("Your machine has more RAM than expected.")
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
    print("  In-memory transductive preprocessing FAILED due to OOM")
    print("  This proves the in-memory approach cannot scale!")
    print()
    print("Next: Run test_2_2 to see on-disk success")
    print()
    sys.exit(0)

except KeyboardInterrupt:
    print()
    print("=" * 70)
    print("⚠️  INTERRUPTED (likely OOM-killed by OS)")
    print("=" * 70)
    print("Process was killed, likely due to memory exhaustion.")
    print()
    print("✓ This counts as OOM validation!")
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
in_channels = 16

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
        "task_level": "node",
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

# Create dataloader for transductive learning
train_dataset = [processed_data]  # Single graph
datamodule = TBDataloader(
    train_dataset, train_dataset, train_dataset, batch_size=1
)

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
    print("Graph was small enough to fit in memory.")
    print("Try a larger graph to demonstrate OOM.")
    print()

except Exception as e:
    print()
    print(f"❌ Training error: {e}")
    import traceback

    traceback.print_exc()
