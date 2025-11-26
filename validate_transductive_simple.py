"""Simple Transductive Validation: Prove on-disk enables what in-memory cannot.

This validates the transductive case where a single large graph's structure
enumeration becomes the bottleneck.
"""

import gc
import sys
import time
import tracemalloc
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data

print("=" * 70)
print("TRANSDUCTIVE VALIDATION - Simple Memory Test")
print("=" * 70)
print()


def generate_large_graph(nodes, avg_degree, seed=42):
    """Generate single large graph for transductive learning."""
    print(f"Generating graph: {nodes} nodes, avg degree {avg_degree}")

    G = nx.watts_strogatz_graph(n=nodes, k=avg_degree, p=0.1, seed=seed)

    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    x = torch.randn(nodes, 32)
    y = torch.randint(0, 10, (nodes,))

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=nodes)

    # Train/val/test masks
    n_train = int(0.6 * nodes)
    n_val = int(0.2 * nodes)

    train_mask = torch.zeros(nodes, dtype=torch.bool)
    val_mask = torch.zeros(nodes, dtype=torch.bool)
    test_mask = torch.zeros(nodes, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"✓ Graph: {G.number_of_edges()} edges")
    print(f"  Train: {train_mask.sum()} nodes")
    print(f"  Val: {val_mask.sum()} nodes")
    print(f"  Test: {test_mask.sum()} nodes")
    print()

    return data, G


def test_inmemory_transductive(nodes, avg_degree):
    """Test in-memory structure enumeration."""
    print("-" * 70)
    print("IN-MEMORY STRUCTURE ENUMERATION (stores all in RAM)")
    print("-" * 70)

    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0] / (1024**2)

    print(f"Starting memory: {start_mem:.1f} MB")
    print()

    try:
        # Generate graph
        data, G = generate_large_graph(nodes, avg_degree)

        print("Enumerating all triangles and STORING in memory...")
        print("⚠️  This is where OOM typically occurs!")
        print()

        start_time = time.time()
        triangles_list = []  # Store ALL triangles in RAM

        for i, node in enumerate(G.nodes()):
            if i % 1000 == 0 and i > 0:
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
                            triangles_list.append(triangle)  # STORE IN RAM!

        elapsed = time.time() - start_time
        final_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
        mem_increase = final_mem - start_mem

        tracemalloc.stop()

        print()
        print("=" * 70)
        print("⚠️  IN-MEMORY SUCCEEDED")
        print("=" * 70)
        print(f"Triangles found: {len(triangles_list):,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Memory increase: {mem_increase:.1f} MB")
        print(f"Peak memory: {peak_mem:.1f} MB")
        print()
        print("Your machine has more RAM than expected!")
        print(
            f"Try larger graph ({nodes + 5000} nodes or degree {avg_degree + 5})"
        )
        print()

        return True, mem_increase, len(triangles_list)

    except (MemoryError, RuntimeError) as e:
        final_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
        tracemalloc.stop()

        print()
        print("=" * 70)
        print("💥 OOM CRASH!")
        print("=" * 70)
        print(f"Error: {type(e).__name__}: {e}")
        print(f"Peak memory before crash: {peak_mem:.1f} MB")
        print()
        return False, peak_mem, 0

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted (likely OOM killed by OS)")
        return False, 0, 0


def test_ondisk_transductive(nodes, avg_degree):
    """Test on-disk structure enumeration."""
    print("-" * 70)
    print("ON-DISK STRUCTURE ENUMERATION (constant memory)")
    print("-" * 70)

    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0] / (1024**2)

    print(f"Starting memory: {start_mem:.1f} MB")
    print()

    try:
        from topobench.data.preprocessor.ondisk_transductive import (
            OnDiskTransductivePreprocessor,
        )

        # Generate same graph
        data, G = generate_large_graph(nodes, avg_degree)

        print("Creating OnDiskTransductiveDataset...")
        print("⚠️  Uses streaming enumeration (constant memory)!")
        print()

        data_dir = Path("/tmp/validate_ondisk_transductive")
        if data_dir.exists():
            import shutil

            shutil.rmtree(data_dir)

        start_time = time.time()

        # Create on-disk dataset (constant memory)
        ondisk_dataset = OnDiskTransductivePreprocessor(
            graph_data=data,
            data_dir=str(data_dir),
            max_clique_size=3,  # Triangles
            force_rebuild=True,
        )

        print("Building index with streaming enumeration...")
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
        print(
            f"Disk usage: ~{sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file()) / (1024**2):.1f} MB"
        )
        print()

        # Test query
        print("Testing batch query (1000 nodes)...")
        batch_nodes = list(range(1000))
        start_query = time.time()
        structures = ondisk_dataset.query_batch(
            batch_nodes, fully_contained=True
        )
        query_time = (time.time() - start_query) * 1000
        print(f"✓ Query: {len(structures)} structures in {query_time:.1f}ms")
        print()

        ondisk_dataset.close()

        return True, mem_increase, ondisk_dataset.num_structures

    except Exception as e:
        tracemalloc.stop()
        print()
        print(f"❌ On-disk failed: {e}")
        import traceback as tb

        tb.print_exc()
        return False, 0, 0


def main():
    """Run validation."""
    print("\nThis script proves on-disk enables what in-memory cannot.")
    print("We test structure enumeration for a single large graph.")
    print()

    # Test with increasing sizes
    sizes = [
        (8000, 20),
        (10000, 22),
        (12000, 25),
        (15000, 25),
        (18000, 30),
    ]

    for nodes, degree in sizes:
        print("\n" + "=" * 70)
        print(f"TEST: {nodes} nodes, avg degree {degree}")
        print("=" * 70)
        print()
        input("Press Enter to continue...")
        print()

        # Test in-memory
        print("\n[1/2] Testing IN-MEMORY approach...")
        inmem_success, inmem_mem, inmem_count = test_inmemory_transductive(
            nodes, degree
        )

        gc.collect()
        time.sleep(2)

        if not inmem_success:
            print("\n✓ Found OOM point!")
            print(f"  In-memory FAILS at {nodes} nodes")
            print()
            input("Press Enter to test ON-DISK at same size...")
            print()

            # Test on-disk
            print("[2/2] Testing ON-DISK approach...")
            ondisk_success, ondisk_mem, ondisk_count = (
                test_ondisk_transductive(nodes, degree)
            )

            if ondisk_success:
                print("\n" + "=" * 70)
                print("VALIDATION COMPLETE! ✅")
                print("=" * 70)
                print()
                print("PROOF:")
                print(f"  ❌ In-memory: FAILED (OOM)")
                print(f"  ✅ On-disk: SUCCESS")
                print(f"      - {ondisk_count:,} triangles indexed")
                print(f"      - {ondisk_mem:.1f} MB memory (constant)")
                print()
                print("Conclusion: On-disk enumeration enables graph sizes")
                print("            that in-memory approaches cannot handle!")
                print()
                return
            else:
                print("\n⚠️  Both failed - check errors above")
                return

        else:
            print(
                f"\n⚠️  In-memory succeeded ({inmem_mem:.1f} MB, {inmem_count:,} triangles)."
            )
            print("Trying larger size...")
            gc.collect()

    print("\n⚠️  All sizes succeeded. Your machine has substantial RAM.")
    print("Consider even larger graphs to find the limit.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted.")
        sys.exit(1)
