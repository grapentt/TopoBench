"""Step 1: Prove in-memory approach fails at scale.

This script demonstrates that traditional in-memory lifting fails
with limited RAM (5GB) at a realistic graph scale.

Expected: OOM crash when trying to store all triangles in RAM.
"""

import gc
import sys
import time
from pathlib import Path

import networkx as nx
import torch
from memory_profiler import profile
from torch_geometric.data import Data


@profile
def test_inmemory_lifting(nodes=12000, avg_degree=25, seed=42):
    """Try to lift large graph in-memory - expected to OOM.

    Parameters
    ----------
    nodes : int
        Number of nodes in graph (12K chosen to OOM on 5GB RAM)
    avg_degree : int
        Average degree (25 creates many triangles)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    int
        Number of triangles found (if doesn't OOM)
    """
    print("=" * 60)
    print("IN-MEMORY LIFTING VALIDATION")
    print("=" * 60)
    print(f"Graph size: {nodes} nodes, avg degree {avg_degree}")
    print(f"Expected memory: ~5-6GB (will likely OOM on 5GB machine)")
    print()

    # Step 1: Generate graph
    print("Step 1/4: Generating Watts-Strogatz graph...")
    start_time = time.time()
    G = nx.watts_strogatz_graph(n=nodes, k=avg_degree, p=0.1, seed=seed)
    num_edges = G.number_of_edges()
    print(
        f"✓ Generated {nodes} nodes, {num_edges} edges ({time.time() - start_time:.1f}s)"
    )
    print()

    # Step 2: Convert to PyG
    print("Step 2/4: Converting to PyTorch Geometric Data...")
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Add reverse edges for undirected graph
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    x = torch.randn(nodes, 32)  # Node features (32 dims)
    y = torch.randint(0, 10, (nodes,))  # Labels for 10 classes
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=nodes)
    print(f"✓ Created PyG Data object")
    print()

    # Step 3: Find and STORE all triangles (THIS SHOULD OOM)
    print("Step 3/4: Finding and storing ALL triangles in RAM...")
    print("⚠️  This is where OOM typically happens with limited RAM!")
    print()

    start_time = time.time()
    triangles = []  # Store ALL triangles in memory (problematic!)

    try:
        for i, node in enumerate(G.nodes()):
            if i % 1000 == 0 and i > 0:
                print(
                    f"   Processed {i}/{nodes} nodes... "
                    f"({len(triangles)} triangles so far)"
                )

            neighbors = list(G.neighbors(node))
            for idx1, n1 in enumerate(neighbors):
                for n2 in neighbors[idx1 + 1 :]:
                    if G.has_edge(n1, n2):
                        triangle = tuple(sorted([node, n1, n2]))
                        if triangle[0] == node:  # Avoid duplicates
                            triangles.append(triangle)  # STORE IN RAM!

        print(
            f"\n✓ Found {len(triangles)} triangles ({time.time() - start_time:.1f}s)"
        )
        print()

        # Step 4: Create lifted features (often OOMs here if step 3 didn't)
        print("Step 4/4: Creating lifted features for all triangles...")
        triangle_features = torch.randn(len(triangles), 64)  # More memory!
        print(f"✓ Created features: {triangle_features.shape}")
        print()

        # If we reach here, it didn't OOM (machine has more RAM than expected)
        print("=" * 60)
        print("⚠️  UNEXPECTED: IN-MEMORY APPROACH SUCCEEDED")
        print("=" * 60)
        print(f"Your machine has more RAM than expected!")
        print(f"Consider increasing graph size to {nodes * 2} nodes.")
        print()

        return len(triangles)

    except MemoryError as e:
        print()
        print("=" * 60)
        print("💥 OOM CRASH (AS EXPECTED)")
        print("=" * 60)
        print(f"MemoryError: {e}")
        print()
        print("This proves that in-memory lifting fails at this scale")
        print("with your available RAM.")
        print()
        raise

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ ERROR: {type(e).__name__}")
        print("=" * 60)
        print(f"{e}")
        print()
        raise


def main():
    """Run validation with increasing sizes until OOM."""
    print("\n" + "=" * 60)
    print("VALIDATION: IN-MEMORY LIFTING FAILURE")
    print("=" * 60)
    print()
    print("This script will attempt to lift a large graph using the")
    print("traditional in-memory approach. With 5GB RAM, we expect")
    print("an out-of-memory (OOM) crash.")
    print()
    print("If it doesn't crash, we'll automatically increase the size.")
    print()
    input("Press Enter to start...")
    print()

    # Try increasing sizes until OOM
    sizes = [8000, 10000, 12000, 15000, 20000]

    for size in sizes:
        try:
            print(f"\nAttempting size: {size} nodes")
            print("-" * 60)
            count = test_inmemory_lifting(nodes=size, avg_degree=25)
            print(f"✅ Size {size}: Success ({count} triangles)")
            print(f"   Trying larger size...")
            print()
            gc.collect()  # Clean up before next attempt

        except MemoryError:
            print()
            print("🎯 FOUND OOM THRESHOLD!")
            print(f"   In-memory fails at {size} nodes")
            print()
            print("VALIDATION RESULT:")
            print(f"  ❌ In-memory lifting: FAILS at {size} nodes")
            print(f"  ✅ Next: Test on-disk approach at same size")
            print()

            # Save result for next script
            result_file = Path("/tmp/oom_threshold.txt")
            result_file.write_text(str(size))
            print(f"Saved OOM threshold to: {result_file}")
            print()
            return size

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            return None

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            return None

    print("\n⚠️  All sizes succeeded (you have more RAM than expected)")
    print("   Consider using size 25000+ for validation")
    return sizes[-1]


if __name__ == "__main__":
    try:
        oom_size = main()
        if oom_size:
            print("\n" + "=" * 60)
            print("NEXT STEP:")
            print("=" * 60)
            print(f"Run: python validate_step2_ondisk_works.py {oom_size}")
            print()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted.")
        sys.exit(1)
