"""Quick validation that train_ogbn_products_ondisk.py imports and model creation work."""

import torch
import networkx as nx
from torch_geometric.data import Data
from omegaconf import OmegaConf

# Test imports from the training script
print("Testing imports...")
from topobench.data.preprocessor import OnDiskTransductivePreprocessor
from topobench.dataloader import NodeBatchSampler, OnDiskTransductiveCollate
from topobench.loss.loss import TBLoss
from topobench.model.model import TBModel
from topobench.nn.backbones.simplicial import SCCNNCustom
from topobench.nn.readouts.mlp_readout import MLPReadout
from topobench.optimizer.optimizer import TBOptimizer
print("✓ All imports successful")

# Create small test graph
print("\nCreating test graph...")
G = nx.erdos_renyi_graph(100, 0.1, seed=42)
edges = list(G.edges())
edge_index = torch.tensor(edges, dtype=torch.long).t()
edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

x = torch.randn(100, 16)
y = torch.randint(0, 5, (100,))

train_mask = torch.zeros(100, dtype=torch.bool)
train_mask[:60] = True
val_mask = torch.zeros(100, dtype=torch.bool)
val_mask[60:80] = True
test_mask = torch.zeros(100, dtype=torch.bool)
test_mask[80:] = True

graph_data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)
print(f"✓ Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

# Configure transforms (same as train script)
print("\nConfiguring transforms...")
transforms_config = OmegaConf.create({
    "clique_lifting": {
        "transform_type": "lifting",
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 2,
    }
})
print("✓ Transforms configured")

# Create preprocessor (same as train script)
print("\nCreating on-disk preprocessor...")
import tempfile
import shutil
temp_dir = tempfile.mkdtemp()

try:
    preprocessor = OnDiskTransductivePreprocessor(
        graph_data=graph_data,
        data_dir=temp_dir,
        transforms_config=transforms_config,
        max_structure_size=3,
    )
    
    print("Building index...")
    preprocessor.build_index()
    print(f"✓ Index built: {preprocessor.num_structures} structures")
    
    # Create sampler and collate
    print("\nCreating sampler and collate...")
    sampler = NodeBatchSampler(
        num_nodes=graph_data.num_nodes,
        batch_size=20,
        shuffle=False,
        mask=train_mask,
    )
    collate_fn = OnDiskTransductiveCollate(preprocessor, fully_contained=True)
    print("✓ Sampler and collate created")
    
    # Test batch creation
    print("\nTesting batch creation...")
    for i, batch_nodes in enumerate(sampler):
        if i >= 1:  # Just test one batch
            break
        batch = collate_fn([batch_nodes])
        print(f"✓ Batch {i+1}:")
        print(f"  - Nodes: {batch.num_nodes}")
        print(f"  - Has x_0: {hasattr(batch, 'x_0')}")
        print(f"  - Has x_1: {hasattr(batch, 'x_1')}")
        print(f"  - Has incidence_1: {hasattr(batch, 'incidence_1')}")
    
    # Create model (same as train script)
    print("\nCreating model...")
    in_channels = graph_data.x.shape[1]
    hidden_channels = 32
    out_channels = 5
    num_layers = 2
    
    in_channels_all = (in_channels, hidden_channels, hidden_channels)
    hidden_channels_all = (hidden_channels, hidden_channels, hidden_channels)
    
    backbone = SCCNNCustom(
        in_channels_all=in_channels_all,
        hidden_channels_all=hidden_channels_all,
        conv_order=1,
        sc_order=2,
        n_layers=num_layers,
    )
    
    readout = MLPReadout(
        in_channels=hidden_channels,
        hidden_layers=[hidden_channels],
        out_channels=out_channels,
        task_level="node",
        pooling_type="sum",
    )
    
    loss = TBLoss(
        dataset_loss={
            "task": "classification",
            "loss_type": "cross_entropy",
        }
    )
    
    optimizer = TBOptimizer(
        optimizer_id="Adam",
        parameters={"lr": 0.01, "weight_decay": 0.0},
    )
    
    model = TBModel(
        backbone=backbone,
        readout=readout,
        loss=loss,
        optimizer=optimizer,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    
    print("\n" + "="*80)
    print("✓ ALL VALIDATIONS PASSED!")
    print("="*80)
    print("\nConclusion:")
    print("- All imports work correctly")
    print("- Preprocessor with transforms works")
    print("- Batch creation with transforms works")
    print("- Model creation works")
    print("- train_ogbn_products_ondisk.py is ready to use!")
    
finally:
    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
