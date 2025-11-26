#!/usr/bin/env python
"""Debug mock dataset with simplicial lifting."""

import torch
from topobench.data.datasets.ogbg_molpcba import MockMolecularDataset
from topobench.transforms.liftings.graph2simplicial import SimplicialCliqueLifting
import networkx as nx
from toponetx.classes import SimplicialComplex

print("=" * 70)
print("DEBUGGING MOCK DATASET + SIMPLICIAL LIFTING")
print("=" * 70)

# Create mock dataset
dataset = MockMolecularDataset(root="./data/debug_mock", num_samples=5, seed=42)

print(f"\n1️⃣ Mock dataset created: {len(dataset)} samples")

# Test first sample
data = dataset[0]
print(f"\n2️⃣ First sample:")
print(f"   Nodes: {data.num_nodes}")
print(f"   edge_index shape: {data.edge_index.shape}")
print(f"   edge_index:\n{data.edge_index}")
print(f"   Has edge_attr: {hasattr(data, 'edge_attr') and data.edge_attr is not None}")

# Simulate _generate_graph_from_data
print(f"\n3️⃣ Simulating _generate_graph_from_data:")
nodes = [(n, dict(features=data.x[n], dim=0)) for n in range(data.x.shape[0])]
edges = [
    (i.item(), j.item(), {})
    for i, j in zip(data.edge_index[0], data.edge_index[1], strict=False)
]

print(f"   Number of edges from edge_index: {len(edges)}")
print(f"   First 10 edges: {edges[:10]}")

# Check for duplicates
edge_set = set()
duplicates = []
for e in edges:
    edge_tuple = tuple(sorted([e[0], e[1]]))
    if edge_tuple in edge_set:
        duplicates.append(e)
    edge_set.add(edge_tuple)

if duplicates:
    print(f"   ⚠️  Found {len(duplicates)} duplicate edges!")
    print(f"   Examples: {duplicates[:5]}")
else:
    print(f"   ✅ No duplicate edges found")

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

print(f"\n4️⃣ NetworkX graph:")
print(f"   Nodes: {G.number_of_nodes()}")
print(f"   Edges: {G.number_of_edges()}")
print(f"   Self-loops: {nx.number_of_selfloops(G)}")

# Try to create simplicial complex
print(f"\n5️⃣ Creating SimplicialComplex...")
try:
    sc = SimplicialComplex(G)
    print(f"   ✅ SUCCESS!")
    print(f"   Number of simplices: {len(sc.simplices)}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Try with lifting transform
print(f"\n6️⃣ Testing with SimplicialCliqueLifting transform...")
try:
    lifting = SimplicialCliqueLifting(complex_dim=2)
    lifted_data = lifting(data)
    print(f"   ✅ SUCCESS!")
    print(f"   Lifted data keys: {lifted_data.keys if hasattr(lifted_data, 'keys') else dir(lifted_data)}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
