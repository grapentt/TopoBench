#!/usr/bin/env python3
"""Research OGB datasets for validation testing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("OGB DATASET RESEARCH")
print("=" * 80)

# ============================================================================
# TRANSDUCTIVE DATASETS (Single Large Graph)
# ============================================================================

print("\n" + "=" * 80)
print("MISSION 2: TRANSDUCTIVE DATASETS (Node Property Prediction)")
print("=" * 80)

transductive_datasets = [
    ('ogbn-products', 'Amazon product co-purchasing network'),
    ('ogbn-proteins', 'Protein-protein association network'),
]

for name, desc in transductive_datasets:
    print(f"\n{'=' * 80}")
    print(f"Dataset: {name}")
    print(f"Description: {desc}")
    print(f"{'=' * 80}")
    
    try:
        from ogb.nodeproppred import NodePropPredDataset
        
        # Load dataset
        print(f"Loading {name}...")
        dataset = NodePropPredDataset(name=name, root='/tmp/ogb_research')
        
        # Get split
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        
        # Get graph
        graph, labels = dataset[0]
        
        print(f"\n✓ Loaded successfully!")
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {graph['num_nodes']:,}")
        print(f"  Edges: {graph['edge_index'].shape[1]:,}")
        
        if 'node_feat' in graph:
            print(f"  Node features: {graph['node_feat'].shape}")
        else:
            print(f"  Node features: None")
        
        if 'edge_feat' in graph:
            print(f"  Edge features: {graph['edge_feat'].shape}")
        else:
            print(f"  Edge features: None")
        
        print(f"\nSplit Statistics:")
        print(f"  Train nodes: {len(train_idx):,}")
        print(f"  Valid nodes: {len(valid_idx):,}")
        print(f"  Test nodes: {len(test_idx):,}")
        
        print(f"\nLabels:")
        print(f"  Shape: {labels.shape}")
        print(f"  Num classes: {dataset.num_classes}")
        
        # Estimate memory for lifting
        num_edges = graph['edge_index'].shape[1]
        avg_degree = num_edges / graph['num_nodes']
        
        print(f"\nComplexity Analysis:")
        print(f"  Average degree: {avg_degree:.1f}")
        print(f"  Estimated triangles: ~{int(num_edges * avg_degree / 6):,} (very rough)")
        
        # Memory estimation
        node_mem_mb = graph['num_nodes'] * 8 / (1024**2)  # 8 bytes per node ID
        edge_mem_mb = num_edges * 16 / (1024**2)  # 2 node IDs per edge
        
        print(f"\nMemory Estimates (Minimum):")
        print(f"  Nodes only: {node_mem_mb:.1f} MB")
        print(f"  Edges only: {edge_mem_mb:.1f} MB")
        print(f"  Graph structure: {node_mem_mb + edge_mem_mb:.1f} MB")
        
        if 'node_feat' in graph:
            feat_mem_mb = graph['node_feat'].numel() * 4 / (1024**2)  # 4 bytes per float
            print(f"  Node features: {feat_mem_mb:.1f} MB")
            print(f"  Total minimum: {node_mem_mb + edge_mem_mb + feat_mem_mb:.1f} MB")
        
        # Suitability assessment
        print(f"\n{'*' * 80}")
        print("SUITABILITY FOR MISSION 2:")
        if graph['num_nodes'] > 100000:
            print("  ✅ EXCELLENT - Large graph (>100K nodes)")
            print("  ✅ Triangle enumeration will be expensive")
            print("  ✅ In-memory lifting likely to OOM or be very slow")
        elif graph['num_nodes'] > 10000:
            print("  ✅ GOOD - Medium-large graph (>10K nodes)")
            print("  ✅ Should demonstrate value of indexing approach")
        else:
            print("  ⚠️  SMALL - May not demonstrate OOM")
        print(f"{'*' * 80}")
        
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# INDUCTIVE DATASETS (Multiple Graphs)
# ============================================================================

print("\n\n" + "=" * 80)
print("MISSION 1: INDUCTIVE DATASETS (Graph Property Prediction)")
print("=" * 80)

inductive_datasets = [
    ('ogbg-molhiv', 'HIV molecule dataset'),
    ('ogbg-molpcba', 'PubChem BioAssay dataset (LARGE)'),
    ('ogbg-ppa', 'Protein-Protein Association graphs'),
]

for name, desc in inductive_datasets:
    print(f"\n{'=' * 80}")
    print(f"Dataset: {name}")
    print(f"Description: {desc}")
    print(f"{'=' * 80}")
    
    try:
        from ogb.graphproppred import GraphPropPredDataset
        
        # Load dataset
        print(f"Loading {name}...")
        dataset = GraphPropPredDataset(name=name, root='/tmp/ogb_research')
        
        # Get split
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        
        print(f"\n✓ Loaded successfully!")
        print(f"\nDataset Statistics:")
        print(f"  Total graphs: {len(dataset):,}")
        print(f"  Train graphs: {len(train_idx):,}")
        print(f"  Valid graphs: {len(valid_idx):,}")
        print(f"  Test graphs: {len(test_idx):,}")
        print(f"  Num tasks: {dataset.num_tasks}")
        
        # Sample a few graphs to get average stats
        sample_indices = [0, len(dataset)//2, len(dataset)-1] if len(dataset) > 2 else [0]
        total_nodes = 0
        total_edges = 0
        
        for idx in sample_indices:
            graph, label = dataset[idx]
            total_nodes += graph['num_nodes']
            total_edges += graph['edge_index'].shape[1]
        
        avg_nodes = total_nodes / len(sample_indices)
        avg_edges = total_edges / len(sample_indices)
        
        print(f"\nGraph Statistics (sampled):")
        print(f"  Average nodes: {avg_nodes:.1f}")
        print(f"  Average edges: {avg_edges:.1f}")
        
        # Get first graph for feature info
        graph, _ = dataset[0]
        if 'node_feat' in graph and graph['node_feat'] is not None:
            print(f"  Node features: {graph['node_feat'].shape[1]} dims")
        else:
            print(f"  Node features: None")
        
        if 'edge_feat' in graph and graph['edge_feat'] is not None:
            print(f"  Edge features: {graph['edge_feat'].shape[1]} dims")
        else:
            print(f"  Edge features: None")
        
        # Memory estimation
        total_nodes_est = len(dataset) * avg_nodes
        total_edges_est = len(dataset) * avg_edges
        
        graph_mem_mb = (total_nodes_est * 8 + total_edges_est * 16) / (1024**2)
        
        print(f"\nMemory Estimates (if all in memory):")
        print(f"  Graph structures: {graph_mem_mb:.1f} MB")
        
        if 'node_feat' in graph and graph['node_feat'] is not None:
            feat_dim = graph['node_feat'].shape[1]
            feat_mem_mb = total_nodes_est * feat_dim * 4 / (1024**2)
            print(f"  Node features: {feat_mem_mb:.1f} MB")
            print(f"  Total minimum: {graph_mem_mb + feat_mem_mb:.1f} MB")
            
            # Estimate after lifting (assuming 2x expansion)
            lifted_mem_mb = (graph_mem_mb + feat_mem_mb) * 2
            print(f"  After lifting (est 2x): {lifted_mem_mb:.1f} MB (~{lifted_mem_mb/1024:.2f} GB)")
        
        # Suitability assessment
        print(f"\n{'*' * 80}")
        print("SUITABILITY FOR MISSION 1:")
        if len(dataset) > 10000:
            print("  ✅ EXCELLENT - Many graphs (>10K)")
            print("  ✅ Lifting all graphs will use significant memory")
            print("  ✅ Strong candidate for OnDisk validation")
        elif len(dataset) > 1000:
            print("  ✅ GOOD - Decent number of graphs (>1K)")
            print("  ✅ Should demonstrate value with lifting")
        else:
            print("  ⚠️  SMALL - May not demonstrate OOM")
        print(f"{'*' * 80}")
        
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n\n" + "█" * 80)
print("RECOMMENDATIONS")
print("█" * 80)

print("\n🎯 MISSION 1 (INDUCTIVE) - Best Candidate:")
print("  Dataset: ogbg-molpcba")
print("  Reason: ~430K molecular graphs")
print("  Challenge: After lifting, InMemory will struggle")
print("  Validation: OnDisk processes one-by-one with constant memory")

print("\n🎯 MISSION 2 (TRANSDUCTIVE) - Best Candidate:")
print("  Dataset: ogbn-products")
print("  Reason: 2.4M nodes, 61M edges")
print("  Challenge: Finding all triangles in such a large graph")
print("  Validation: Offline indexing vs in-memory enumeration")

print("\n" + "█" * 80)
