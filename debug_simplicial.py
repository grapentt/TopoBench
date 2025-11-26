#!/usr/bin/env python
"""Debug simplicial lifting issue."""

import networkx as nx
import torch
from torch_geometric.data import Data
from toponetx.classes import SimplicialComplex

# Create a simple clean graph
num_nodes = 5
G = nx.connected_watts_strogatz_graph(num_nodes, k=2, p=0.3, seed=42)

print("=" * 70)
print("DEBUGGING TOPONETX SIMPLICIAL COMPLEX")
print("=" * 70)

print("\n1️⃣ NetworkX Graph:")
print(f"   Nodes: {list(G.nodes())}")
print(f"   Edges: {list(G.edges())}")
print(f"   Has self-loops: {nx.number_of_selfloops(G)}")
print(f"   Is multigraph: {isinstance(G, nx.MultiGraph)}")

# Try to create SimplicialComplex
print("\n2️⃣ Attempting to create SimplicialComplex from graph...")
try:
    sc = SimplicialComplex(G)
    print(f"   ✅ SUCCESS!")
    print(f"   Simplices: {sc.simplices}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Now try with PyG data (single direction edges)
print("\n3️⃣ Creating PyG Data with SINGLE-direction edges:")
edges = list(G.edges())
edges_normalized = [(min(u, v), max(u, v)) for u, v in edges]
edges_unique = list(set(edges_normalized))
edge_index = torch.tensor(edges_unique, dtype=torch.long).t()

print(f"   edge_index shape: {edge_index.shape}")
print(f"   edge_index:\n{edge_index}")

x = torch.randn(num_nodes, 9)
data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

print("\n4️⃣ Converting PyG Data to NetworkX (simulating _generate_graph_from_data):")
nodes = [(n, dict(features=data.x[n], dim=0)) for n in range(data.x.shape[0])]
edges_from_pyg = [
    (i.item(), j.item(), {})
    for i, j in zip(data.edge_index[0], data.edge_index[1], strict=False)
]

print(f"   Nodes: {[n[0] for n in nodes]}")
print(f"   Edges: {edges_from_pyg}")

G2 = nx.Graph()
G2.add_nodes_from(nodes)
G2.add_edges_from(edges_from_pyg)

print(f"   NetworkX graph created")
print(f"   G2 nodes: {list(G2.nodes())}")
print(f"   G2 edges: {list(G2.edges())}")

print("\n5️⃣ Attempting to create SimplicialComplex from PyG-derived graph...")
try:
    sc2 = SimplicialComplex(G2)
    print(f"   ✅ SUCCESS!")
    print(f"   Simplices: {sc2.simplices}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
