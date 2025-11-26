#!/usr/bin/env python
"""Test if real OGB data has proper edge format for simplicial lifting."""

import sys

try:
    from ogb.graphproppred import PygGraphPropPredDataset
    import torch
    
    print("=" * 70)
    print("TESTING REAL OGB EDGE FORMAT")
    print("=" * 70)
    
    # Load real OGB dataset (just to check format, no training)
    print("\n1️⃣ Loading OGB dataset (this will download if needed)...")
    print("   Note: This is ~500 MB download on first run")
    
    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root="./data/ogb_test")
    
    print(f"   ✅ Loaded: {len(dataset)} total graphs")
    
    # Check first sample
    print("\n2️⃣ Checking first sample...")
    data = dataset[0]
    
    print(f"   Nodes: {data.num_nodes}")
    print(f"   edge_index shape: {data.edge_index.shape}")
    print(f"   y shape: {data.y.shape}")
    print(f"   First 10 edges:\n{data.edge_index[:, :10]}")
    
    # Check if edges are bidirectional
    print("\n3️⃣ Analyzing edge direction...")
    edge_set_fwd = set()
    edge_set_bwd = set()
    
    for i in range(min(100, data.edge_index.shape[1])):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        edge = (min(src, dst), max(src, dst))
        if src < dst:
            edge_set_fwd.add(edge)
        elif src > dst:
            edge_set_bwd.add(edge)
    
    print(f"   Forward edges (src < dst): {len(edge_set_fwd)}")
    print(f"   Backward edges (dst < src): {len(edge_set_bwd)}")
    print(f"   Both directions: {len(edge_set_fwd & edge_set_bwd)}")
    
    if edge_set_fwd & edge_set_bwd:
        print(f"\n   ⚠️  OGB data has BIDIRECTIONAL edges!")
        print(f"   This will cause SimplicalComplex 'duplicate nodes' error")
        print(f"   We need to filter to single-direction edges")
    else:
        print(f"\n   ✅ OGB data has single-direction edges only!")
        print(f"   Ready for simplicial lifting!")
    
    # Check for self-loops
    self_loops = sum(1 for i in range(data.edge_index.shape[1]) 
                     if data.edge_index[0, i] == data.edge_index[1, i])
    print(f"\n4️⃣ Self-loops: {self_loops}")
    if self_loops > 0:
        print(f"   ⚠️  Has self-loops (will cause issues)")
    else:
        print(f"   ✅ No self-loops")
    
    print("\n" + "=" * 70)
    
except ImportError:
    print("❌ OGB not installed. Run: pip install ogb")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
