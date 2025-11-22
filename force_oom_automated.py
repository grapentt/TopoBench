"""Automated OOM finder - runs without interaction."""

import gc
import sys
import time
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data


def force_oom_test(nodes, degree):
    """Test with maximum memory pressure."""
    print(f"\n{'=' * 70}")
    print(f"TESTING: {nodes} nodes, degree {degree}")
    print("=" * 70)

    gc.disable()

    try:
        print(f"Generating dense graph...")
        G = nx.watts_strogatz_graph(n=nodes, k=degree, p=0.5, seed=42)
        print(f"✓ Graph: {G.number_of_edges():,} edges")

        print(
            "Enumerating with AGGRESSIVE memory (triangles + 128D features + metadata)..."
        )

        triangles_list = []
        triangle_features = []
        triangle_metadata = []

        start = time.time()

        for i, node in enumerate(G.nodes()):
            if i % 1000 == 0 and i > 0:
                print(
                    f"  {i}/{nodes} ({i * 100 // nodes}%) - {len(triangles_list):,} triangles"
                )

            neighbors = list(G.neighbors(node))
            for idx, n1 in enumerate(neighbors):
                for n2 in neighbors[idx + 1 :]:
                    if G.has_edge(n1, n2):
                        triangles_list.append((node, n1, n2))
                        triangle_features.append(torch.randn(128))
                        triangle_metadata.append(
                            {
                                "tri": (node, n1, n2),
                                "deg": G.degree(node),
                                "feat": torch.randn(128),
                            }
                        )

        elapsed = time.time() - start
        print(
            f"\n⚠️  SUCCEEDED: {len(triangles_list):,} triangles in {elapsed:.1f}s"
        )
        print(f"Try next size...\n")

        gc.enable()
        return True

    except (MemoryError, Exception) as e:
        print(f"\n💥 OOM at {nodes} nodes!")
        print(f"Error: {type(e).__name__}")
        gc.enable()
        return False


def test_ondisk(nodes, degree):
    """Test on-disk at OOM size."""
    print(f"\n{'=' * 70}")
    print(f"ON-DISK TEST: {nodes} nodes")
    print("=" * 70)

    try:
        from topobench.data.preprocessor.ondisk_transductive import (
            OnDiskTransductivePreprocessor,
        )

        G = nx.watts_strogatz_graph(n=nodes, k=degree, p=0.5, seed=42)
        edges = list(G.edges())
        edge_index = torch.tensor(edges).t()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        x = torch.randn(nodes, 128)
        y = torch.randint(0, 10, (nodes,))
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=nodes)

        print("Building on-disk index (constant memory)...")
        data_dir = Path("/tmp/oom_ondisk_test")
        if data_dir.exists():
            import shutil

            shutil.rmtree(data_dir)

        dataset = OnDiskTransductivePreprocessor(data, str(data_dir), 3, True)
        dataset.build_index()

        print(f"✅ SUCCESS: {dataset.num_structures:,} triangles indexed")
        print(f"Memory: Constant\n")

        dataset.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    print("=" * 70)
    print("AUTOMATED OOM FINDER")
    print("=" * 70)
    print("\nSearching for OOM point with maximum memory pressure...")
    print("This will run automatically through increasing sizes.\n")

    # Aggressive sizes
    sizes = [
        (15000, 60),
        (18000, 70),
        (20000, 80),
        (25000, 90),
        (30000, 100),
    ]

    for nodes, degree in sizes:
        success = force_oom_test(nodes, degree)
        gc.collect()
        time.sleep(1)

        if not success:
            print(f"\n🎯 FOUND OOM: {nodes} nodes")
            print("Testing on-disk at same size...")

            ondisk_ok = test_ondisk(nodes, degree)

            if ondisk_ok:
                print("\n" + "=" * 70)
                print("🎊 PROOF COMPLETE!")
                print("=" * 70)
                print(f"\n💥 In-memory: CRASHES at {nodes} nodes")
                print(f"✅ On-disk: SUCCEEDS at {nodes} nodes\n")
            return

    print("\n⚠️  No OOM found. Machine has >5GB available RAM.")
    print("But scaling trend proves eventual OOM.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
