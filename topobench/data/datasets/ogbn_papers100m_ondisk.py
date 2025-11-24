"""On-disk dataset for OGBN-Papers100M with memory-efficient node sampling.

This implementation demonstrates handling massive single-graph datasets using
BaseOnDiskInductiveDataset by treating node neighborhoods as individual samples.
"""

import os.path as osp
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch_geometric.data import Data

from topobench.data.datasets.base_inductive import OnDemandInductiveDataset


class OGBNPapers100MOnDiskDataset(OnDemandInductiveDataset):
    """Memory-efficient dataset for OGBN-Papers100M.

    This dataset demonstrates our on-disk approach for massive graphs by:
    1. Storing graph structure and features on disk (memory-mapped)
    2. Generating node-centric subgraphs on-demand
    3. Using lightweight pickling for parallel data loading

    Instead of loading the entire 111M node graph into RAM, we:
    - Store edge_index and node features as memory-mapped files
    - Generate k-hop neighborhoods around target nodes on-demand
    - Cache frequently accessed subgraphs

    Parameters
    ----------
    root : str or Path
        Root directory where the dataset will be saved.
    num_samples : int, optional
        Number of node samples to use from the dataset.
        Use fewer for quick demos, all 111M for full experiments.
        Default: 10000 (suitable for demonstration).
    k_hop : int, optional
        Number of hops for neighborhood sampling (default: 2).
    cache_samples : bool, optional
        Enable caching of generated subgraphs (default: True).

    Example
    -------
    >>> # Quick demo with 10K nodes
    >>> dataset = OGBNPapers100MOnDiskDataset(
    ...     root="/data/papers100m",
    ...     num_samples=10000,
    ...     k_hop=2
    ... )
    >>> # Lightweight pickling for parallel loading
    >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    >>>
    >>> # Full dataset (111M nodes)
    >>> full_dataset = OGBNPapers100MOnDiskDataset(
    ...     root="/data/papers100m",
    ...     num_samples=111059956,  # All nodes
    ...     k_hop=2
    ... )

    Notes
    -----
    - First run downloads and processes the dataset (one-time cost)
    - Graph structure stored as memory-mapped numpy arrays
    - Node features stored as memory-mapped numpy arrays
    - Each __getitem__ call generates a fresh subgraph (O(1) memory)
    - Suitable for full-graph training with mini-batch sampling
    """

    URLS: ClassVar = {
        "ogbn-papers100M": "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
    }

    def __init__(
        self,
        root: str | Path,
        num_samples: int = 10000,
        k_hop: int = 2,
        cache_samples: bool = True,
        seed: int = 42,
    ):
        """Initialize OGBN-Papers100M on-disk dataset.

        Parameters
        ----------
        root : str or Path
            Root directory for dataset storage.
        num_samples : int, optional
            Number of node samples (default: 10000 for demo).
        k_hop : int, optional
            Neighborhood hops (default: 2).
        cache_samples : bool, optional
            Enable subgraph caching (default: True).
        seed : int, optional
            Random seed for reproducibility (default: 42).
        """
        self.dataset_root = Path(root) / "ogbn_papers100M"
        self.num_samples_param = num_samples
        self.k_hop = k_hop

        # Set up directory structure
        self.raw_dir = self.dataset_root / "raw"
        self.processed_dir = self.dataset_root / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Download and process if needed
        if not self._is_processed():
            self._download_and_process()

        # Load memory-mapped arrays (doesn't load into RAM!)
        self._load_memmap()

        # Initialize parent class
        super().__init__(
            root=self.processed_dir,
            num_samples=num_samples,
            cache_samples=cache_samples,
            seed=seed,
        )

    def _is_processed(self) -> bool:
        """Check if dataset is already processed."""
        required_files = [
            "edge_index.npy",
            "node_feat.npy",
            "node_label.npy",
            "num_nodes.txt",
        ]
        return all((self.processed_dir / f).exists() for f in required_files)

    def _download_and_process(self):
        """Download and process OGBN-Papers100M dataset.

        This function:
        1. Downloads raw data using OGB
        2. Converts to memory-mapped numpy arrays
        3. Stores metadata
        """
        print("Downloading and processing OGBN-Papers100M...")
        print("This is a one-time operation and may take some time.")

        try:
            from ogb.nodeproppred import NodePropPredDataset
        except ImportError as e:
            raise ImportError(
                "ogb package required. Install with: pip install ogb"
            ) from e

        # Patch torch.load for PyTorch 2.6+ compatibility
        original_load = torch.load

        def patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load

        try:
            # Download using OGB (will load into memory temporarily)
            print("Downloading from OGB...")
            dataset = NodePropPredDataset(
                name="ogbn-papers100M", root=str(self.dataset_root / "ogb_raw")
            )
            graph, labels = dataset[0]

            print("Converting to memory-mapped format...")

            # Save edge_index as memory-mapped array
            edge_index = graph["edge_index"]
            np.save(self.processed_dir / "edge_index.npy", edge_index)

            # Save node features as memory-mapped array
            node_feat = graph["node_feat"]
            np.save(self.processed_dir / "node_feat.npy", node_feat)

            # Save labels
            np.save(self.processed_dir / "node_label.npy", labels)

            # Save metadata
            num_nodes = graph["num_nodes"]
            with open(self.processed_dir / "num_nodes.txt", "w") as f:
                f.write(str(num_nodes))

            print("✓ Processing complete!")
            print(f"  Nodes: {num_nodes:,}")
            print(f"  Edges: {edge_index.shape[1]:,}")
            print(f"  Features: {node_feat.shape[1]}")

        finally:
            # Restore original torch.load
            torch.load = original_load

    def _load_memmap(self):
        """Load memory-mapped arrays (doesn't load into RAM!)."""
        # Memory-mapped arrays: accessing elements loads only needed data
        self.edge_index_mmap = np.load(
            self.processed_dir / "edge_index.npy", mmap_mode="r"
        )
        self.node_feat_mmap = np.load(
            self.processed_dir / "node_feat.npy", mmap_mode="r"
        )
        self.node_label_mmap = np.load(
            self.processed_dir / "node_label.npy", mmap_mode="r"
        )

        with open(self.processed_dir / "num_nodes.txt") as f:
            self.num_nodes_total = int(f.read().strip())

    def _generate_sample(self, idx: int, seed: int) -> Data:
        """Generate a k-hop neighborhood subgraph around node idx.

        This is called by GeneratedInductiveDataset when needed.
        Each call loads only the necessary data from memory-mapped arrays.

        Parameters
        ----------
        idx : int
            Target node index.
        seed : int
            Random seed (for reproducibility).

        Returns
        -------
        Data
            PyG Data object with k-hop neighborhood.
        """
        # For demo: use idx as the target node
        # For full training: you'd want more sophisticated sampling
        target_node = idx % self.num_nodes_total

        # Simple k-hop neighborhood (for demonstration)
        # In practice, you'd use PyG's NeighborSampler for efficiency
        if self.k_hop == 0:
            # Just the node itself
            node_ids = np.array([target_node])
            edge_mask = np.zeros(self.edge_index_mmap.shape[1], dtype=bool)
        else:
            # Sample k-hop neighborhood
            # This is simplified; use PyG's k_hop_subgraph for production
            node_ids, edge_index, _, _ = self._simple_k_hop_subgraph(
                target_node, self.k_hop
            )

        # Load features only for sampled nodes (O(k) memory, not O(n)!)
        x = torch.from_numpy(self.node_feat_mmap[node_ids].copy())
        y = torch.from_numpy(self.node_label_mmap[node_ids].copy())

        # Create subgraph
        if self.k_hop > 0:
            edge_index_tensor = torch.from_numpy(edge_index)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

        return Data(
            x=x,
            y=y,
            edge_index=edge_index_tensor,
            num_nodes=len(node_ids),
            target_node_idx=torch.tensor([0]),  # Target is first node after reindexing
        )

    def _simple_k_hop_subgraph(self, node_idx: int, k: int):
        """Simple k-hop neighborhood extraction.

        Note: This is a simplified implementation for demonstration.
        For production, use torch_geometric.utils.k_hop_subgraph.
        """
        # Start with target node
        current_nodes = {node_idx}
        all_nodes = {node_idx}
        edges = []

        # Expand k times
        for _ in range(k):
            new_nodes = set()
            for node in current_nodes:
                # Find edges involving this node (simplified)
                # In practice, you'd use an adjacency list for efficiency
                edge_mask = (self.edge_index_mmap[0] == node) | (
                    self.edge_index_mmap[1] == node
                )
                node_edges = self.edge_index_mmap[:, edge_mask]

                if node_edges.shape[1] > 0:
                    edges.append(node_edges)
                    neighbors = np.unique(node_edges)
                    new_nodes.update(neighbors)

            all_nodes.update(new_nodes)
            current_nodes = new_nodes

            if not current_nodes:
                break

        # Combine edges
        if edges:
            edge_index = np.hstack(edges)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)

        # Reindex nodes to [0, num_nodes)
        node_ids = np.array(sorted(all_nodes))
        node_mapping = {old: new for new, old in enumerate(node_ids)}

        # Reindex edges
        if edge_index.shape[1] > 0:
            edge_index_reindexed = np.array(
                [[node_mapping[edge_index[0, i]], node_mapping[edge_index[1, i]]]
                 for i in range(edge_index.shape[1])
                 if edge_index[0, i] in node_mapping and edge_index[1, i] in node_mapping]
            ).T
        else:
            edge_index_reindexed = edge_index

        return node_ids, edge_index_reindexed, None, None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"total_nodes={self.num_nodes_total:,}, "
            f"num_samples={len(self)}, "
            f"k_hop={self.k_hop})"
        )

    @property
    def num_features(self) -> int:
        """Number of node features."""
        return self.node_feat_mmap.shape[1]

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return int(self.node_label_mmap.max()) + 1
