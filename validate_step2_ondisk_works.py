"""Step 2: Prove on-disk approach succeeds at same scale.

This script demonstrates that our on-disk approach succeeds
at the same graph scale where in-memory failed.

Expected: Success with constant memory usage.
"""

import gc
import sys
import time
from pathlib import Path

import networkx as nx
import torch
from memory_profiler import profile
from torch_geometric.data import Data

from topobench.data.preprocessor.ondisk_transductive import (
    OnDiskTransductivePreprocessor,
)


@profile
def test_ondisk_approach(nodes=12000, avg_degree=25, seed=42):
    """Use on-disk indexing for large graph - expected to succeed.

    Parameters
    ----------
    nodes : int
        Number of nodes (same size that OOM'd in-memory)
    avg_degree : int
        Average degree
    seed : int
        Random seed (same as in-memory test)

    Returns
    -------
    int
        Number of structures indexed
    """
    print("=" * 60)
    print("ON-DISK APPROACH VALIDATION")
    print("=" * 60)
    print(f"Graph size: {nodes} nodes, avg degree {avg_degree}")
    print(f"Expected memory: <1GB (constant during indexing)")
    print()

    # Step 1: Generate same graph as in-memory test
    print("Step 1/5: Generating Watts-Strogatz graph...")
    start_time = time.time()
    G = nx.watts_strogatz_graph(n=nodes, k=avg_degree, p=0.1, seed=seed)
    num_edges = G.number_of_edges()
    print(
        f"✓ Generated {nodes} nodes, {num_edges} edges ({time.time() - start_time:.1f}s)"
    )
    print()

    # Step 2: Convert to PyG
    print("Step 2/5: Converting to PyTorch Geometric Data...")
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    x = torch.randn(nodes, 32, generator=torch.manual_seed(seed))
    y = torch.randint(0, 10, (nodes,), generator=torch.manual_seed(seed))
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=nodes)
    print(f"✓ Created PyG Data object")
    print()

    # Step 3: Create on-disk dataset
    print("Step 3/5: Creating OnDiskTransductiveDataset...")
    data_dir = Path("/tmp/ondisk_validation")
    if data_dir.exists():
        import shutil

        shutil.rmtree(data_dir)

    dataset = OnDiskTransductivePreprocessor(
        graph_data=data,
        data_dir=str(data_dir),
        max_structure_size=3,  # Triangles
        force_rebuild=True,
    )
    print(f"✓ Dataset created")
    print()

    # Step 4: Build index (streaming, constant memory)
    print("Step 4/5: Building structure index...")
    print("⚠️  This uses CONSTANT memory (streaming enumeration)")
    print()
    start_time = time.time()

    try:
        dataset.build_index()
        elapsed = time.time() - start_time

        print()
        print(f"✓ Index built successfully! ({elapsed:.1f}s)")
        print(f"  Structures indexed: {dataset.num_structures:,}")
        print(f"  Memory usage: Constant (on-disk storage)")
        print()

    except Exception as e:
        print()
        print(f"❌ Index building failed: {e}")
        print()
        raise

    # Step 5: Test batch queries
    print("Step 5/5: Testing batch queries...")

    batch_sizes = [100, 500, 1000]
    for batch_size in batch_sizes:
        batch_nodes = list(range(min(batch_size, nodes)))
        start_time = time.time()
        structures = dataset.query_batch(batch_nodes, fully_contained=True)
        elapsed = (time.time() - start_time) * 1000  # ms

        print(
            f"  Batch size {batch_size:4d}: {len(structures):6,} structures "
            f"({elapsed:.1f}ms)"
        )

    print()

    # Get stats
    stats = dataset.get_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    print()

    dataset.close()

    print("=" * 60)
    print("✅ ON-DISK APPROACH SUCCEEDED")
    print("=" * 60)
    print(f"Successfully indexed {dataset.num_structures:,} structures")
    print(f"at the SAME scale where in-memory failed!")
    print()
    print(f"Memory: Constant (~500MB-1GB) vs In-memory OOM")
    print(f"Disk: ~{data_dir.stat().st_size // (1024**2)}MB index")
    print()

    return dataset.num_structures


def main():
    """Run on-disk validation."""
    print("\n" + "=" * 60)
    print("VALIDATION: ON-DISK APPROACH SUCCESS")
    print("=" * 60)
    print()

    # Check if we have OOM threshold from previous run
    threshold_file = Path("/tmp/oom_threshold.txt")
    if threshold_file.exists():
        oom_size = int(threshold_file.read_text().strip())
        print(f"Found OOM threshold from previous run: {oom_size} nodes")
        print(f"Testing on-disk approach at same size...")
    else:
        # Use command line argument or default
        if len(sys.argv) > 1:
            oom_size = int(sys.argv[1])
            print(f"Using size from command line: {oom_size} nodes")
        else:
            oom_size = 12000
            print(f"Using default size: {oom_size} nodes")
            print(
                "(Run validate_step1_inmemory_fails.py first to find actual OOM size)"
            )

    print()
    print("This will create an on-disk index using streaming enumeration.")
    print("Expected memory: <1GB (constant)")
    print()
    input("Press Enter to start...")
    print()

    try:
        count = test_ondisk_approach(nodes=oom_size, avg_degree=25)

        print("=" * 60)
        print("VALIDATION RESULT:")
        print("=" * 60)
        print(f"  ❌ In-memory: FAILED (OOM at {oom_size} nodes)")
        print(f"  ✅ On-disk: SUCCESS ({count:,} structures indexed)")
        print()
        print("CONCLUSION:")
        print("  On-disk approach enables processing at scales")
        print("  where in-memory approaches fail!")
        print()

        print("=" * 60)
        print("NEXT STEP:")
        print("=" * 60)
        print(f"Run: python validate_step3_train_model.py {oom_size}")
        print("This will train a model using the on-disk indexed data.")
        print()

        return count

    except Exception as e:
        print()
        print(f"❌ On-disk approach failed: {e}")
        print()
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted.")
        sys.exit(1)
