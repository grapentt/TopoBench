#!/usr/bin/env python
"""Test if caching modifies the data."""

from topobench.data.datasets.ogbg_molpcba import MockMolecularDataset
from pathlib import Path
import shutil

print("=" * 70)
print("TESTING IF CACHING MODIFIES DATA")
print("=" * 70)

# Clean cache
cache_dir = Path("./data/test_cache")
if cache_dir.exists():
    shutil.rmtree(cache_dir)

# Create dataset with caching
print("\n1️⃣ Creating dataset with caching enabled...")
dataset = MockMolecularDataset(root=cache_dir, num_samples=3, seed=42, cache_samples=True)

# Load sample 0 first time
print("\n2️⃣ Loading sample 0 (first time, will cache)...")
data1 = dataset[0]
print(f"   edge_index shape: {data1.edge_index.shape}")
print(f"   edge_index:\n{data1.edge_index[:, :5]}")  # First 5 edges

# Load sample 0 second time (from cache)
print("\n3️⃣ Loading sample 0 (second time, from cache)...")
data2 = dataset[0]
print(f"   edge_index shape: {data2.edge_index.shape}")
print(f"   edge_index:\n{data2.edge_index[:, :5]}")  # First 5 edges

# Check if they're identical
print("\n4️⃣ Comparing...")
if data1.edge_index.shape == data2.edge_index.shape:
    if (data1.edge_index == data2.edge_index).all():
        print(f"   ✅ Identical!")
    else:
        print(f"   ⚠️  Different values!")
        print(f"   Difference: {(data1.edge_index != data2.edge_index).sum()} elements")
else:
    print(f"   ❌ Different shapes!")
    print(f"   First: {data1.edge_index.shape}, Second: {data2.edge_index.shape}")

# Check edge directions
def analyze_edges(edge_index, name):
    edges_fwd = 0
    edges_bwd = 0
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src < dst:
            edges_fwd += 1
        else:
            edges_bwd += 1
    print(f"\n{name}:")
    print(f"   Forward (src < dst): {edges_fwd}")
    print(f"   Backward (src > dst): {edges_bwd}")
    return edges_fwd, edges_bwd

print("\n5️⃣ Edge direction analysis:")
fwd1, bwd1 = analyze_edges(data1.edge_index, "First load")
fwd2, bwd2 = analyze_edges(data2.edge_index, "Second load (cached)")

if fwd1 == fwd2 and bwd1 == bwd2:
    print(f"\n✅ Edge directions preserved through caching!")
else:
    print(f"\n⚠️  Edge directions changed!")

print("=" * 70)
