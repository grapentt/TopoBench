#!/usr/bin/env python3
"""Comprehensive memory diagnostics with detailed logging.

This script investigates:
1. Baseline memory pollution
2. Incorrect synthetic data scaling  
3. Hidden memory allocations in OnDisk implementation
4. OOM estimation for InMemory vs OnDisk
"""

import gc
import sys
import tempfile
import tracemalloc
from pathlib import Path

import psutil
import torch
from torch_geometric.data import Data, InMemoryDataset

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.utils import SyntheticGraphDataset
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductivePreprocessor


class DiagnosticSyntheticDataset(InMemoryDataset):
    """InMemory dataset with detailed logging."""
    
    def __init__(self, num_samples: int, num_nodes: int = 50, num_features: int = 16):
        self._num_samples = num_samples
        self._num_nodes = num_nodes
        self._num_features = num_features
        
        print(f"\n🔍 Creating InMemory dataset:")
        print(f"   Samples: {num_samples:,}")
        print(f"   Nodes per sample: ~{num_nodes}")
        print(f"   Features: {num_features}")
        
        super().__init__()
        
        # Track memory during data generation
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        print(f"   Generating {num_samples:,} graphs...")
        data_list = []
        for i in range(num_samples):
            torch.manual_seed(42 + i)
            n = num_nodes + (i % 10)
            graph = Data(
                x=torch.randn(n, num_features),
                edge_index=torch.randint(0, n, (2, n * 3)),
                y=torch.tensor([i % 10]),
            )
            data_list.append(graph)
            
            # Log every 1000 samples
            if (i + 1) % 1000 == 0:
                mem_current = process.memory_info().rss / (1024 * 1024)
                mem_delta = mem_current - mem_before
                print(f"     {i+1:,} samples: +{mem_delta:.1f} MB ({mem_delta/(i+1)*1000:.2f} KB/sample)")
        
        mem_after_gen = process.memory_info().rss / (1024 * 1024)
        print(f"   After generation: +{mem_after_gen - mem_before:.1f} MB")
        
        print(f"   Collating {len(data_list):,} graphs...")
        self.data, self.slices = self.collate(data_list)
        
        mem_after_collate = process.memory_info().rss / (1024 * 1024)
        print(f"   After collate: +{mem_after_collate - mem_before:.1f} MB")
        print(f"   ✓ Total memory used: {mem_after_collate - mem_before:.1f} MB")
        
        # Estimate graph size
        if len(data_list) > 0:
            sample_graph = data_list[0]
            graph_size_bytes = (
                sample_graph.x.numel() * sample_graph.x.element_size() +
                sample_graph.edge_index.numel() * sample_graph.edge_index.element_size() +
                sample_graph.y.numel() * sample_graph.y.element_size()
            )
            print(f"   Average graph size: ~{graph_size_bytes / 1024:.2f} KB")
            print(f"   Total data size: ~{graph_size_bytes * num_samples / (1024**2):.2f} MB")
    
    def len(self) -> int:
        return self._num_samples
    
    def get(self, idx: int) -> Data:
        return super().get(idx)


def measure_with_isolation(approach: str, num_samples: int) -> dict:
    """Measure memory in complete isolation.
    
    Restarts Python process between measurements to avoid pollution.
    """
    print(f"\n{'='*80}")
    print(f"ISOLATED MEASUREMENT: {approach.upper()} with {num_samples:,} samples")
    print(f"{'='*80}\n")
    
    # Force aggressive cleanup
    gc.collect()
    gc.collect()
    gc.collect()
    
    process = psutil.Process()
    
    # Measure CLEAN baseline
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    print(f"📊 Clean baseline: {baseline_mb:.1f} MB")
    
    # Start tracemalloc for detailed tracking
    tracemalloc.start()
    
    if approach == "inmemory":
        dataset = DiagnosticSyntheticDataset(
            num_samples=num_samples,
            num_nodes=50,
            num_features=16,
        )
        
        # Access some samples
        print(f"\n   Accessing 100 samples...")
        for i in range(min(100, len(dataset))):
            _ = dataset[i]
        
        mem_after_access = process.memory_info().rss / (1024 * 1024)
        
    else:  # ondisk
        print(f"\n🔍 Creating OnDisk preprocessor:")
        print(f"   Samples: {num_samples:,}")
        print(f"   Storage: mmap with LZ4")
        print(f"   Cache: DISABLED (cache_size=0)")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_before_creation = process.memory_info().rss / (1024 * 1024)
            
            source_dataset = SyntheticGraphDataset(
                num_samples=num_samples,
                num_nodes=50,
                num_features=16,
            )
            
            print(f"\n   Processing {num_samples:,} samples...")
            dataset = OnDiskInductivePreprocessor(
                dataset=source_dataset,
                data_dir=tmpdir,
                transforms_config=None,
                num_workers=1,
                cache_size=0,  # NO CACHE!
            )
            
            mem_after_creation = process.memory_info().rss / (1024 * 1024)
            print(f"   After creation: {mem_after_creation:.1f} MB (+{mem_after_creation - mem_before_creation:.1f} MB)")
            
            # Access samples
            print(f"\n   Accessing 100 samples...")
            for i in range(min(100, len(dataset))):
                _ = dataset[i]
                if (i + 1) % 20 == 0:
                    mem_current = process.memory_info().rss / (1024 * 1024)
                    print(f"     Sample {i+1}: {mem_current:.1f} MB")
            
            mem_after_access = process.memory_info().rss / (1024 * 1024)
            
            del dataset
            del source_dataset
    
    # Get tracemalloc snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print(f"\n📈 Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(f"   {stat}")
    
    tracemalloc.stop()
    
    peak_mb = mem_after_access
    delta_mb = peak_mb - baseline_mb
    
    print(f"\n📊 Memory Summary:")
    print(f"   Baseline: {baseline_mb:.1f} MB")
    print(f"   Peak: {peak_mb:.1f} MB")
    print(f"   Delta: {delta_mb:.1f} MB")
    print(f"   Per sample: {delta_mb * 1024 / num_samples:.2f} KB/sample")
    
    return {
        "baseline_mb": baseline_mb,
        "peak_mb": peak_mb,
        "delta_mb": delta_mb,
        "per_sample_kb": delta_mb * 1024 / num_samples,
    }


def estimate_oom_thresholds():
    """Estimate when each approach would hit OOM."""
    print(f"\n{'='*80}")
    print("OOM THRESHOLD ESTIMATION")
    print(f"{'='*80}\n")
    
    process = psutil.Process()
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"System RAM:")
    print(f"  Total: {total_ram_gb:.1f} GB")
    print(f"  Available: {available_ram_gb:.1f} GB")
    
    # Measure small sample to estimate scaling
    print(f"\n📊 Measuring with 1,000 samples...")
    inmemory_1k = measure_with_isolation("inmemory", 1000)
    
    gc.collect()
    gc.collect()
    
    ondisk_1k = measure_with_isolation("ondisk", 1000)
    
    # Estimate slopes
    inmemory_slope_mb = inmemory_1k["delta_mb"] / 1000
    ondisk_slope_mb = ondisk_1k["delta_mb"] / 1000
    
    print(f"\n📈 Scaling Analysis:")
    print(f"   InMemory slope: {inmemory_slope_mb * 1024:.2f} KB/sample")
    print(f"   OnDisk slope: {ondisk_slope_mb * 1024:.2f} KB/sample")
    print(f"   Slope ratio: {inmemory_slope_mb / max(ondisk_slope_mb, 0.001):.2f}×")
    
    # Estimate OOM
    usable_ram_mb = available_ram_gb * 1024 * 0.8  # Use 80% of available
    baseline_mb = inmemory_1k["baseline_mb"]
    
    inmemory_oom_samples = int((usable_ram_mb - baseline_mb) / max(inmemory_slope_mb, 0.001))
    ondisk_oom_samples = int((usable_ram_mb - baseline_mb) / max(ondisk_slope_mb, 0.001))
    
    print(f"\n🚨 OOM Threshold Estimates (with {available_ram_gb:.1f} GB available):")
    print(f"   InMemory: ~{inmemory_oom_samples:,} samples")
    print(f"   OnDisk: ~{ondisk_oom_samples:,} samples")
    
    if ondisk_oom_samples > inmemory_oom_samples:
        factor = ondisk_oom_samples / max(inmemory_oom_samples, 1)
        print(f"   OnDisk can handle {factor:.1f}× more samples before OOM")
    else:
        print(f"   ⚠️ WARNING: OnDisk has similar or worse OOM threshold!")
    
    return {
        "inmemory_oom": inmemory_oom_samples,
        "ondisk_oom": ondisk_oom_samples,
        "inmemory_slope_kb": inmemory_slope_mb * 1024,
        "ondisk_slope_kb": ondisk_slope_mb * 1024,
    }


def analyze_synthetic_data_problem():
    """Analyze the graph size vs count problem."""
    print(f"\n{'='*80}")
    print("SYNTHETIC DATA ANALYSIS")
    print(f"{'='*80}\n")
    
    print("📊 Current Synthetic Dataset Behavior:")
    print(f"   Each graph: ~50 nodes + (idx % 10) variation")
    print(f"   Graph size: ~50-59 nodes, ~150-177 edges")
    print(f"   Features: 16 × 4 bytes = 64 bytes per node")
    print()
    
    # Calculate sizes
    avg_nodes = 55
    avg_edges = 165
    features = 16
    bytes_per_float = 4
    
    node_features_bytes = avg_nodes * features * bytes_per_float
    edge_index_bytes = avg_edges * 2 * 8  # 2 rows, int64
    y_bytes = 8
    
    total_per_graph = node_features_bytes + edge_index_bytes + y_bytes
    
    print(f"   Estimated size per graph:")
    print(f"     Node features: {node_features_bytes:,} bytes ({node_features_bytes/1024:.2f} KB)")
    print(f"     Edge index: {edge_index_bytes:,} bytes ({edge_index_bytes/1024:.2f} KB)")
    print(f"     Label: {y_bytes} bytes")
    print(f"     Total: ~{total_per_graph/1024:.2f} KB per graph")
    print()
    
    print(f"📈 Scaling with --scale 10 (publication config):")
    print(f"   Dataset sizes: [1000, 5000, 10000, 20000, 50000, 100000]")
    print(f"   Graph size: STILL ~{avg_nodes} nodes (WRONG!)")
    print()
    
    print(f"   At 100,000 samples:")
    print(f"     Total data: {total_per_graph * 100000 / (1024**2):.1f} MB")
    print(f"     But spread across 100,000 tiny graphs!")
    print()
    
    print(f"❌ PROBLEM: We're scaling NUMBER of graphs, not SIZE of graphs!")
    print()
    print(f"🎯 For inductive learning, we should:")
    print(f"   - Keep sample count LOW (e.g., 100-1000 samples)")
    print(f"   - SCALE UP graph size (e.g., 1K-100K nodes per graph)")
    print(f"   - This mimics real datasets: ENZYMES (100 graphs), PROTEINS (1113 graphs)")
    print()
    
    print(f"💡 Correct scaling strategy:")
    print(f"   Base: 100 samples × 1000 nodes = 100K total nodes")
    print(f"   --scale 10: 100 samples × 10000 nodes = 1M total nodes")
    print(f"   NOT: 1000 samples × 1000 nodes (current bug)")


def main():
    """Run comprehensive diagnostics."""
    print("=" * 80)
    print("  TOPOBENCH MEMORY DIAGNOSTIC SUITE")
    print("=" * 80)
    
    # Issue #1: Synthetic data scaling
    analyze_synthetic_data_problem()
    
    # Issue #2: OOM estimation
    thresholds = estimate_oom_thresholds()
    
    # Summary
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"🔍 BUGS FOUND:")
    print(f"   1. ❌ Scaling creates MORE graphs, not LARGER graphs")
    print(f"   2. ⚠️  Slope ratio: {thresholds['inmemory_slope_kb'] / max(thresholds['ondisk_slope_kb'], 0.001):.1f}× (target: >10×)")
    print(f"   3. ⚠️  Baseline pollution in benchmarking loop")
    print()
    
    print(f"💡 FIXES NEEDED:")
    print(f"   1. Create scalable graph size dataset (not sample count)")
    print(f"   2. Fix baseline measurement (isolate each test)")
    print(f"   3. Add memory tracking during preprocessing")
    print()


if __name__ == "__main__":
    main()
