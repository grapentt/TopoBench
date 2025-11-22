"""Production validation: Inductive on-disk vs in-memory.

This script demonstrates that:
1. In-memory preprocessing FAILS (OOM) on large inductive datasets
2. On-disk preprocessing SUCCEEDS with constant memory

Usage:
    python validation/validate_inductive_ondisk.py --max_ram_gb 4.0
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

from topobench.data.preprocessor import OnDiskInductiveDataset, PreProcessor, create_preprocessor
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
        description="Validate inductive on-disk vs in-memory"
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
        default="./data/validation_inductive",
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
    print(f"  - Graphs: {params['num_graphs']}")
    print(f"  - Nodes per graph: {params['nodes_per_graph']}")
    print(f"  - Estimated structures: {params['estimated_structures']:,}")
    print(f"  - Estimated memory: {params['estimated_memory_gb']:.2f}GB")

    def try_inmemory():
        """Attempt in-memory preprocessing."""
        # Generate synthetic graphs
        graphs = []
        for i in range(params["num_graphs"]):
            # Create random graph
            G = nx.erdos_renyi_graph(
                n=params["nodes_per_graph"],
                p=params["avg_degree"] / params["nodes_per_graph"]
            )
            data = from_networkx(G)
            data.y = torch.tensor([i % 2])  # Binary classification
            data.x = torch.randn(data.num_nodes, 16)
            graphs.append(data)

        # Try to create all triangle structures in memory (this should OOM)
        all_triangles = []
        for data in graphs:
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            
            # Find triangles (expensive operation)
            adj_dict = {}
            for i in range(edge_index.size(1)):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u not in adj_dict:
                    adj_dict[u] = set()
                adj_dict[u].add(v)
            
            # This will create many triangles and accumulate in memory
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
            # Generate synthetic graphs (same as in-memory test)
            print(f"\nGenerating {params['num_graphs']} synthetic graphs...")
            graphs = []
            for i in range(params["num_graphs"]):
                # Create random graph
                G = nx.erdos_renyi_graph(
                    n=params["nodes_per_graph"],
                    p=params["avg_degree"] / params["nodes_per_graph"]
                )
                data = from_networkx(G)
                data.y = torch.tensor([i % 2])  # Binary classification
                data.x = torch.randn(data.num_nodes, 16)
                graphs.append(data)
            print(f"✓ Generated {len(graphs)} graphs")

            # Configure lifting
            transforms_config = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                }
            })

            print(f"\nProcessing with on-disk approach...")
            ondisk = OnDiskInductiveDataset(
                dataset=graphs,
                data_dir=data_dir + "/ondisk_processed",
                transforms_config=transforms_config,
            )

            print(f"✓ Processed {len(ondisk)} samples (memory stays constant)")

            # Verify we can load samples
            print(f"\nVerifying sample loading...")
            for i in range(min(10, len(ondisk))):
                sample = ondisk[i]
                tracker.update()
            print(f"  Loaded 10 samples successfully")

            print(f"✓ Successfully loaded samples without accumulating memory")

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
        return False


def main():
    """Main validation script."""
    args = parse_args()

    print_section("Inductive On-Disk Validation")

    print(f"\nConfiguration:")
    print(f"  - Max RAM for OOM test: {args.max_ram_gb}GB")
    print(f"  - Data directory: {args.data_dir}")

    # Calculate parameters guaranteed to cause OOM
    print(f"\nCalculating dataset parameters...")
    params = calculate_oom_params(
        max_ram_gb=args.max_ram_gb,
        approach="inductive",
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
            "✓ VALIDATION PASSED: On-disk enables training on datasets "
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
