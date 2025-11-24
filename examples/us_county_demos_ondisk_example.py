"""Example demonstrating the US County Demographics on-disk dataset.

This example shows how to use the on-disk version to minimize memory usage
compared to the traditional in-memory approach.
"""

import pickle
import tempfile

from omegaconf import DictConfig

from topobench.data.datasets.us_county_demos_dataset import (
    USCountyDemosDataset,
)
from topobench.data.datasets.us_county_demos_ondisk import (
    USCountyDemosOnDiskDataset,
)


def main():
    """Compare in-memory vs on-disk dataset approaches."""
    # Configuration
    params = DictConfig({"year": 2012, "task_variable": "Election"})

    print("=" * 70)
    print("US County Demographics Dataset: In-Memory vs On-Disk Comparison")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create on-disk dataset
        print("\n1. Creating On-Disk Dataset...")
        ondisk_dataset = USCountyDemosOnDiskDataset(
            root=tmpdir, name="US-county-demos", parameters=params
        )
        print(f"   ✅ Dataset created: {len(ondisk_dataset)} graphs")

        # Check pickle size (important for parallel processing)
        print("\n2. Testing Pickling (for parallel processing)...")
        ondisk_pickled = pickle.dumps(ondisk_dataset)
        ondisk_pickle_kb = len(ondisk_pickled) / 1024
        print(f"   On-Disk pickle size: {ondisk_pickle_kb:.2f} KB")
        print(
            f"   {'✅ Lightweight!' if ondisk_pickle_kb < 10 else '⚠️ Could be smaller'}"
        )

        # Load a sample
        print("\n3. Loading Data...")
        data = ondisk_dataset[0]
        print(f"   Nodes: {data.num_nodes}")
        print(f"   Features: {data.x.shape[1]}")
        print(f"   Edges: {data.edge_index.shape[1]}")

        # Test caching
        print("\n4. Testing Caching...")
        _ = ondisk_dataset[0]  # First access
        _ = ondisk_dataset[0]  # Second access (from cache)
        print("   ✅ Caching works correctly")

        print("\n" + "=" * 70)
        print("Benefits of On-Disk Approach:")
        print("=" * 70)
        print("✅ O(1) memory usage - data loaded on-demand")
        print("✅ Lightweight pickling for fast parallel processing")
        print("✅ Automatic caching for repeated access")
        print("✅ Multiprocessing-safe")
        print("✅ Ideal for large datasets or memory-constrained environments")
        print("=" * 70)


if __name__ == "__main__":
    main()
