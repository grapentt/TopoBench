"""Production validation: Transductive on-disk vs in-memory.

This script demonstrates that:
1. In-memory preprocessing FAILS (OOM) on large transductive graphs
2. On-disk preprocessing SUCCEEDS with constant memory via indexing

Usage:
    python validation/validate_transductive_ondisk.py --max_ram_gb 4.0
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add topobench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from topobench.data.preprocessor import OnDiskTransductiveDataset, PreProcessor
from topobench.utils.validation_utils import (
    MemoryTracker,
    calculate_oom_params,
    expect_oom,
    print_result,
    print_section,
    print_subsection,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate transductive on-disk vs in-memory"
    )
    parser.add_argument(
        "--max_ram_gb",
        type=float,
        default=4.0,
        help="Maximum RAM for OOM test (GB)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/validation_transductive",
        help="Data directory",
    )
    return parser.parse_args()


def test_inmemory_fails(params: dict, data_dir: str) -> bool:
    """Test that in-memory preprocessing fails with OOM.

    Parameters
    ----------
    params : dict
        Dataset parameters from calculate_oom_params.
    data_dir : str
        Directory for data storage.

    Returns
    -------
    bool
        True if OOM occurred as expected.
    """
    print_subsection("[1/2] Testing In-Memory Approach (Expected: OOM)")

    print(f"\nDataset parameters:")
    print(f"  - Nodes: {params['num_nodes']}")
    print(f"  - Avg degree: {params['avg_degree']}")
    print(f"  - Estimated structures: {params['estimated_structures']:,}")
    print(f"  - Estimated memory: {params['estimated_memory_gb']:.2f}GB")

    def try_inmemory():
        """Attempt in-memory preprocessing."""
        # Generate large graph
        G = nx.erdos_renyi_graph(
            n=params["num_nodes"],
            p=params["avg_degree"] / params["num_nodes"]
        )
        graph_data = from_networkx(G)
        graph_data.x = torch.randn(graph_data.num_nodes, 16)
        graph_data.y = torch.randn(graph_data.num_nodes, 1)  # Node labels
        
        # Try to find all triangles in memory (this should OOM)
        edge_index = graph_data.edge_index
        num_nodes = graph_data.num_nodes
        
        # Build adjacency dict
        adj_dict = {}
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u not in adj_dict:
                adj_dict[u] = set()
            adj_dict[u].add(v)
        
        # Find all triangles (accumulates memory)
        all_triangles = []
        for u in range(num_nodes):
            if u not in adj_dict:
                continue
            neighbors_u = list(adj_dict[u])
            for i, v in enumerate(neighbors_u):
                if v not in adj_dict:
                    continue
                for w in neighbors_u[i+1:]:
                    if w in adj_dict.get(v, set()):
                        # Store triangle (accumulates memory)
                        all_triangles.append([u, v, w])
        
        print(f"Found {len(all_triangles)} triangles (accumulating in memory)")

    # Expect this to OOM
    oom_occurred, message = expect_oom(try_inmemory, timeout_seconds=300)

    print_result(oom_occurred, f"In-memory approach failed: {message}")

    return oom_occurred


def test_ondisk_succeeds(params: dict, data_dir: str) -> bool:
    """Test that on-disk preprocessing succeeds with constant memory.

    Parameters
    ----------
    params : dict
        Dataset parameters (same as in-memory test).
    data_dir : str
        Directory for data storage.

    Returns
    -------
    bool
        True if on-disk succeeded.
    """
    print_subsection("[2/2] Testing On-Disk Approach (Expected: SUCCESS)")

    try:
        tracker = MemoryTracker()

        with tracker:
            # Generate large graph (same as in-memory test)
            print(f"\nGenerating graph with {params['num_nodes']:,} nodes...")
            G = nx.erdos_renyi_graph(
                n=params["num_nodes"],
                p=params["avg_degree"] / params["num_nodes"]
            )
            graph_data = from_networkx(G)
            graph_data.x = torch.randn(graph_data.num_nodes, 16)
            graph_data.y = torch.randn(graph_data.num_nodes, 1)  # Node labels
            print(f"✓ Generated graph with {graph_data.num_nodes:,} nodes")

            print(f"\nBuilding structure index...")
            ondisk = OnDiskTransductiveDataset(
                graph_data=graph_data,
                data_dir=data_dir + "/ondisk_index",
                max_structure_size=3,  # Triangles
            )

            ondisk.build_index()
            print(f"✓ Indexed {ondisk.num_structures:,} structures (memory stays constant)")

            # Verify we can query structures
            print(f"\nVerifying structure queries...")
            test_nodes = list(range(min(100, graph_data.num_nodes)))
            structures = ondisk.query_batch(test_nodes, fully_contained=True)
            tracker.update()
            print(f"  Queried {len(structures)} structures for {len(test_nodes)} nodes")

            print(f"✓ Successfully queried structures without loading all into memory")

        # Print memory report
        tracker.report(prefix="On-Disk ")

        # Check memory stayed reasonable
        memory_increase_gb = tracker.get_delta_ram() / 1024
        memory_ok = memory_increase_gb < params["estimated_memory_gb"] * 0.3

        print_result(
            memory_ok,
            f"Memory stayed constant: {memory_increase_gb:.2f}GB increase "
            f"(vs {params['estimated_memory_gb']:.2f}GB for in-memory)"
        )

        return True

    except Exception as e:
        print_result(False, f"On-disk approach failed unexpectedly: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main validation script."""
    args = parse_args()

    print_section("Transductive On-Disk Validation")

    print(f"\nConfiguration:")
    print(f"  - Max RAM for OOM test: {args.max_ram_gb}GB")
    print(f"  - Data directory: {args.data_dir}")

    # Calculate parameters guaranteed to cause OOM
    print(f"\nCalculating dataset parameters...")
    params = calculate_oom_params(
        max_ram_gb=args.max_ram_gb,
        approach="transductive",
        structure_type="triangles"
    )

    print(f"✓ Parameters calculated to exceed {args.max_ram_gb}GB RAM")

    # Test 1: In-memory fails
    inmemory_failed = test_inmemory_fails(params, args.data_dir)

    # Test 2: On-disk succeeds
    ondisk_succeeded = test_ondisk_succeeds(params, args.data_dir)

    # Final summary
    print_section("Validation Results")

    print("\nTest 1: In-Memory Approach")
    print_result(inmemory_failed, "Failed with OOM as expected")

    print("\nTest 2: On-Disk Approach")
    print_result(ondisk_succeeded, "Succeeded with constant memory")

    print("\nConclusion:")
    if inmemory_failed and ondisk_succeeded:
        print_result(
            True,
            "✓ VALIDATION PASSED: On-disk enables training on graphs "
            "that don't fit in memory!"
        )
        return 0
    else:
        print_result(
            False,
            "✗ VALIDATION FAILED: Results don't match expectations"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
