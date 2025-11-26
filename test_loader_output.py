#!/usr/bin/env python
"""Test what the loader returns."""

from omegaconf import OmegaConf
from topobench.data.loaders.ogbg_molpcba_loader import OGBGMolPCBALoader
from topobench.transforms.liftings.graph2simplicial import SimplicialCliqueLifting

print("=" * 70)
print("TESTING LOADER OUTPUT")
print("=" * 70)

# Create loader with same config as experiment
config = OmegaConf.create({
    "data_dir": "./data/test_loader_output",
    "subset_size": 10,  # Test with 10 samples
    "split": "train",
    "use_mock": True,
})

loader = OGBGMolPCBALoader(config)
dataset, data_dir = loader.load()

print(f"\n1️⃣ Loader returned dataset: {type(dataset).__name__}")
print(f"   Length: {len(dataset)}")

# Test first sample
data = dataset[0]
print(f"\n2️⃣ First sample from loader:")
print(f"   Nodes: {data.num_nodes}")
print(f"   edge_index shape: {data.edge_index.shape}")
print(f"   edge_index:\n{data.edge_index}")

# Check for bidirectional edges
edges_fwd = set()
edges_bwd = set()
for i in range(data.edge_index.shape[1]):
    src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
    edge = (min(src, dst), max(src, dst))
    if src < dst:
        edges_fwd.add(edge)
    else:
        edges_bwd.add(edge)

print(f"\n3️⃣ Edge direction analysis:")
print(f"   Forward edges (src < dst): {len(edges_fwd)}")
print(f"   Backward edges (dst < src): {len(edges_bwd)}")
print(f"   Bidirectional: {len(edges_fwd & edges_bwd)}")

if edges_fwd & edges_bwd:
    print(f"   ⚠️  FOUND BIDIRECTIONAL EDGES!")
else:
    print(f"   ✅ Only single-direction edges")

# Try lifting
print(f"\n4️⃣ Testing simplicial lifting...")
try:
    lifting = SimplicialCliqueLifting(complex_dim=2)
    lifted_data = lifting(data)
    print(f"   ✅ SUCCESS!")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("=" * 70)
