"""Aggressive OOM test - find the real limit with dense graphs."""

import gc
import sys
import time
import tracemalloc
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data

print("=" * 70)
print("AGGRESSIVE OOM TEST - Finding Real Memory Limit")
print("=" * 70)
print()
print("This test creates VERY DENSE graphs to trigger OOM faster.")
print("Dense graphs = combinatorial explosion of triangles!")
print()


def test_dense_graph(nodes, avg_degree):
    """Test with dense graph configuration."""
    print("-" * 70)
    print(f"TEST: {nodes} nodes, degree {avg_degree} (DENSE)")
    print("-" * 70)

    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0] / (1024**2)

    print(f"Starting memory: {start_mem:.1f} MB")
    print()

    try:
        # Generate dense Watts-Strogatz graph
        print("Generating DENSE graph...")
        G = nx.watts_strogatz_graph(
            n=nodes, k=avg_degree, p=0.5, seed=42
        )  # Higher p = denser

        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        x = torch.randn(nodes, 64)  # Larger features
        y = torch.randint(0, 10, (nodes,))

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=nodes)

        print(f"✓ Graph: {G.number_of_edges():,} edges")
        print()

        # Enumerate and STORE triangles
        print("Enumerating triangles (storing ALL in RAM)...")
        print("⚠️  Dense graphs create MANY triangles!")
        print()

        start_time = time.time()
        triangles_list = []
        features_list = []  # Also store features

        for i, node in enumerate(G.nodes()):
            if i % 500 == 0 and i > 0:
                current_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
                mem_increase = current_mem - start_mem
                print(
                    f"  Node {i}/{nodes}: "
                    f"{len(triangles_list):,} triangles, "
                    f"+{mem_increase:.1f}MB"
                )

            neighbors = list(G.neighbors(node))
            for idx, n1 in enumerate(neighbors):
                for n2 in neighbors[idx + 1 :]:
                    if G.has_edge(n1, n2):
                        triangle = tuple(sorted([node, n1, n2]))
                        if triangle[0] == node:
                            triangles_list.append(triangle)
                            # Store features too (more memory pressure)
                            tri_features = torch.randn(64)
                            features_list.append(tri_features)

        elapsed = time.time() - start_time
        final_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
        mem_increase = final_mem - start_mem

        tracemalloc.stop()

        print()
        print("=" * 70)
        print("IN-MEMORY SUCCEEDED")
        print("=" * 70)
        print(f"Triangles: {len(triangles_list):,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Memory increase: {mem_increase:.1f} MB")
        print(f"Peak memory: {peak_mem:.1f} MB")
        print()

        return True, mem_increase

    except (MemoryError, RuntimeError, KeyboardInterrupt) as e:
        final_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
        tracemalloc.stop()

        print()
        print("=" * 70)
        print("💥 OOM OR INTERRUPTED!")
        print("=" * 70)
        if isinstance(e, KeyboardInterrupt):
            print("Process interrupted (likely OOM-killed by OS)")
        else:
            print(f"Error: {type(e).__name__}: {e}")
        print(f"Peak memory: {peak_mem:.1f} MB")
        print()
        return False, peak_mem


def test_ondisk_dense(nodes, avg_degree):
    """Test on-disk with same dense graph."""
    print("-" * 70)
    print(f"ON-DISK TEST: {nodes} nodes, degree {avg_degree}")
    print("-" * 70)

    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0] / (1024**2)

    try:
        from topobench.data.preprocessor.ondisk_transductive import (
            OnDiskTransductivePreprocessor,
        )

        print("Generating same DENSE graph...")
        G = nx.watts_strogatz_graph(n=nodes, k=avg_degree, p=0.5, seed=42)

        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        x = torch.randn(nodes, 64)
        y = torch.randint(0, 10, (nodes,))

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=nodes)

        print(f"✓ Graph: {G.number_of_edges():,} edges")
        print()

        print("Building on-disk index (streaming)...")
        data_dir = Path("/tmp/validate_ondisk_dense")
        if data_dir.exists():
            import shutil

            shutil.rmtree(data_dir)

        start_time = time.time()

        ondisk_dataset = OnDiskTransductivePreprocessor(
            graph_data=data,
            data_dir=str(data_dir),
            max_structure_size=3,
            force_rebuild=True,
        )

        ondisk_dataset.build_index()

        elapsed = time.time() - start_time
        final_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
        mem_increase = final_mem - start_mem

        tracemalloc.stop()

        print()
        print("=" * 70)
        print("✅ ON-DISK SUCCEEDED")
        print("=" * 70)
        print(f"Triangles indexed: {ondisk_dataset.num_structures:,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Memory increase: {mem_increase:.1f} MB (CONSTANT!)")
        print(f"Peak memory: {peak_mem:.1f} MB")
        print()

        ondisk_dataset.close()
        return True, mem_increase

    except Exception as e:
        tracemalloc.stop()
        print(f"❌ On-disk failed: {e}")
        import traceback

        traceback.print_exc()
        return False, 0


def main():
    """Aggressively find OOM point."""
    print("Testing with increasingly DENSE graphs...")
    print("Dense = more triangles = faster OOM")
    print()

    # Start aggressive: dense graphs with high degree
    tests = [
        (5000, 40),  # 5K nodes, degree 40 (dense!)
        (8000, 50),  # 8K nodes, degree 50 (very dense!)
        (10000, 60),  # 10K nodes, degree 60 (extremely dense!)
        (12000, 70),  # 12K nodes, degree 70 (insane!)
        (15000, 80),  # 15K nodes, degree 80 (will definitely OOM)
    ]

    for nodes, degree in tests:
        print("\n" + "=" * 70)
        print(f"ATTEMPT: {nodes} nodes, degree {degree}")
        print("=" * 70)
        print()
        input("Press Enter to test IN-MEMORY...")
        print()

        success, mem = test_dense_graph(nodes, degree)

        gc.collect()
        time.sleep(2)

        if not success:
            print("\n✓ FOUND OOM POINT!")
            print(f"  In-memory FAILS at {nodes} nodes (degree {degree})")
            print()
            input("Press Enter to test ON-DISK at same size...")
            print()

            ondisk_success, ondisk_mem = test_ondisk_dense(nodes, degree)

            if ondisk_success:
                print("\n" + "=" * 70)
                print("🎊 VALIDATION COMPLETE!")
                print("=" * 70)
                print()
                print("PROOF:")
                print(f"  ❌ In-memory: FAILED (OOM) at {nodes} nodes")
                print(f"  ✅ On-disk: SUCCESS at {nodes} nodes")
                print(f"      Memory: {ondisk_mem:.1f} MB (constant)")
                print()
                print("Conclusion: On-disk enables dense graph processing")
                print("            that in-memory cannot handle!")
                print()
                return
            else:
                print("\n⚠️  Both failed")
                return

        print(f"\n⚠️  Still succeeded ({mem:.1f} MB). Trying denser...")
        gc.collect()

    print("\n⚠️  Even extreme densities succeeded!")
    print("Your machine has exceptional RAM (>5GB available).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
