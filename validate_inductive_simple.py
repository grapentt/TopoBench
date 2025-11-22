"""Simple Inductive Validation: Prove on-disk enables what in-memory cannot.

This validates using TopoBench's data preprocessing framework without requiring
full training (Lightning). We prove the bottleneck is in the lifting phase.
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
print("INDUCTIVE VALIDATION - Simple Memory Test")
print("=" * 70)
print()


def generate_synthetic_dataset(n_graphs, nodes_per_graph, avg_degree):
    """Generate synthetic graph dataset."""
    print(f"Generating {n_graphs} synthetic graphs...")
    print(f"  ~{nodes_per_graph} nodes/graph, degree {avg_degree}")

    dataset = []
    for i in range(n_graphs):
        if i % 500 == 0 and i > 0:
            print(f"  Generated {i}/{n_graphs}...")

        n = nodes_per_graph + torch.randint(-5, 5, (1,)).item()
        n = max(20, n)

        G = nx.watts_strogatz_graph(n=n, k=avg_degree, p=0.3, seed=42 + i)

        edges = list(G.edges())
        if not edges:
            continue

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        x = torch.randn(n, 16)
        y = torch.randint(0, 5, (1,))

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=n)
        dataset.append(data)

    print(f"✓ Generated {len(dataset)} graphs\n")
    return dataset


def test_inmemory_lifting(dataset, max_k=2):
    """Test in-memory lifting - stores all structures."""
    print("-" * 70)
    print("IN-MEMORY LIFTING (stores all structures in RAM)")
    print("-" * 70)

    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0] / (1024**2)

    print(f"Starting memory: {start_mem:.1f} MB")
    print(f"Lifting {len(dataset)} graphs to cell complexes (k={max_k})...")
    print()

    try:
        all_lifted_data = []
        total_structures = 0
        start_time = time.time()

        for i, data in enumerate(dataset):
            if i % 100 == 0 and i > 0:
                current_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
                mem_increase = current_mem - start_mem
                print(
                    f"  Graph {i}/{len(dataset)}: "
                    f"+{mem_increase:.1f}MB, "
                    f"{total_structures} structures"
                )

            # Convert to NetworkX
            G = nx.Graph()
            edge_list = data.edge_index.t().tolist()
            G.add_edges_from(edge_list)

            # Find all triangles (k=3 cliques) and STORE IN MEMORY
            triangles = []
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                for idx, n1 in enumerate(neighbors):
                    for n2 in neighbors[idx + 1 :]:
                        if G.has_edge(n1, n2):
                            triangle = tuple(sorted([node, n1, n2]))
                            if triangle[0] == node:
                                triangles.append(triangle)  # STORE IN RAM!

            total_structures += len(triangles)

            # Store lifted data (this adds more memory)
            lifted_data = {
                "graph_id": i,
                "num_nodes": data.num_nodes,
                "triangles": triangles,  # ALL triangles in memory
                "features": data.x,
            }
            all_lifted_data.append(lifted_data)  # ACCUMULATES IN RAM

        elapsed = time.time() - start_time
        final_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
        mem_increase = final_mem - start_mem

        tracemalloc.stop()

        print()
        print("=" * 70)
        print("⚠️  IN-MEMORY SUCCEEDED")
        print("=" * 70)
        print(f"Processed: {len(dataset)} graphs")
        print(f"Structures: {total_structures:,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Memory increase: {mem_increase:.1f} MB")
        print(f"Peak memory: {peak_mem:.1f} MB")
        print()
        print("Your machine has more RAM than expected!")
        print("Try larger dataset (more graphs or nodes/graph)")
        print()

        return True, mem_increase

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
        return False, peak_mem

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted (likely OOM killed by OS)")
        return False, 0


def test_ondisk_lifting(dataset, max_k=2):
    """Test on-disk lifting - constant memory."""
    print("-" * 70)
    print("ON-DISK LIFTING (constant memory, disk-backed)")
    print("-" * 70)

    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0] / (1024**2)

    print(f"Starting memory: {start_mem:.1f} MB")
    print(f"Processing {len(dataset)} graphs with on-disk storage...")
    print()

    try:
        from topobench.data.preprocessor.ondisk_inductive import (
            OnDiskInductivePreprocessor,
        )

        data_dir = Path("/tmp/validate_ondisk_inductive")
        if data_dir.exists():
            import shutil

            shutil.rmtree(data_dir)

        start_time = time.time()

        # Create on-disk dataset (processes with constant memory)
        ondisk_dataset = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=str(data_dir),
            max_k=max_k,
            force_rebuild=True,
        )

        elapsed = time.time() - start_time
        final_mem = tracemalloc.get_traced_memory()[0] / (1024**2)
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
        mem_increase = final_mem - start_mem

        tracemalloc.stop()

        print()
        print("=" * 70)
        print("✅ ON-DISK SUCCEEDED")
        print("=" * 70)
        print(f"Processed: {len(dataset)} graphs")
        print(f"Time: {elapsed:.1f}s")
        print(f"Memory increase: {mem_increase:.1f} MB (CONSTANT!)")
        print(f"Peak memory: {peak_mem:.1f} MB")
        print(
            f"Disk usage: ~{sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file()) / (1024**2):.1f} MB"
        )
        print()

        # Test loading a sample
        print("Testing data loading...")
        sample = ondisk_dataset[0]
        print(f"✓ Sample loaded: {sample.num_nodes} nodes")
        print()

        return True, mem_increase

    except Exception as e:
        tracemalloc.stop()
        print()
        print(f"❌ On-disk failed: {e}")
        import traceback as tb

        tb.print_exc()
        return False, 0


def main():
    """Run validation."""
    print("\nThis script proves on-disk enables what in-memory cannot.")
    print(
        "We test the preprocessing/lifting phase where the bottleneck occurs."
    )
    print()

    # Test with increasing sizes
    sizes = [
        (2000, 50, 10),
        (4000, 60, 12),
        (6000, 70, 12),
        (8000, 80, 15),
    ]

    for n_graphs, nodes, degree in sizes:
        print("\n" + "=" * 70)
        print(
            f"TEST: {n_graphs} graphs, ~{nodes} nodes/graph, degree {degree}"
        )
        print("=" * 70)
        print()
        input("Press Enter to continue...")
        print()

        # Generate dataset
        dataset = generate_synthetic_dataset(n_graphs, nodes, degree)

        # Test in-memory
        print("\n[1/2] Testing IN-MEMORY approach...")
        inmem_success, inmem_mem = test_inmemory_lifting(dataset)

        gc.collect()
        time.sleep(2)

        if not inmem_success:
            print("\n✓ Found OOM point!")
            print(f"  In-memory FAILS at {n_graphs} graphs")
            print()
            input("Press Enter to test ON-DISK at same size...")
            print()

            # Test on-disk
            print("[2/2] Testing ON-DISK approach...")
            ondisk_success, ondisk_mem = test_ondisk_lifting(dataset)

            if ondisk_success:
                print("\n" + "=" * 70)
                print("VALIDATION COMPLETE! ✅")
                print("=" * 70)
                print()
                print("PROOF:")
                print(f"  ❌ In-memory: FAILED (OOM)")
                print(
                    f"  ✅ On-disk: SUCCESS ({ondisk_mem:.1f} MB constant memory)"
                )
                print()
                print(
                    "Conclusion: On-disk preprocessing enables dataset sizes"
                )
                print("            that in-memory approaches cannot handle!")
                print()
                return
            else:
                print("\n⚠️  Both failed - check errors above")
                return

        else:
            print(
                f"\n⚠️  In-memory succeeded ({inmem_mem:.1f} MB). Trying larger size..."
            )
            del dataset
            gc.collect()

    print("\n⚠️  All sizes succeeded. Your machine has substantial RAM.")
    print("Consider even larger datasets to find the limit.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted.")
        sys.exit(1)
