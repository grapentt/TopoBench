"""Loader for OGBN-products dataset.

This module provides a loader for the OGBN-products dataset, which is a large-scale
product co-purchasing network from Amazon with 2.4M nodes and 61M edges.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from omegaconf import DictConfig

from topobench.data.datasets.mock_transductive_graph import MockTransductiveGraph
from topobench.data.loaders.base import AbstractLoader


class OGBNProductsLoader(AbstractLoader):
    """Loader for OGBN-products dataset.

    OGBN-products is a large-scale undirected graph representing Amazon products
    where nodes are products and edges indicate they are purchased together.
    The task is to predict the product category (47 classes).

    Dataset Statistics:
    - Nodes: 2,449,029
    - Edges: 61,859,140 (undirected)
    - Node features: 100-dimensional
    - Classes: 47 product categories
    - Avg degree: ~50

    This is an excellent dataset for testing on-disk transductive learning since:
    - Too large for most RAM (structures would be ~10-30GB in memory)
    - High degree → many triangles
    - Real-world graph topology

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters including:
        - data_dir: Directory for storing the dataset
        - data_name: Name identifier (default: "ogbn-products")
        - subset_nodes: Optional, sample subgraph with N nodes (default: None = full graph)
        - use_mock: If True, use MockTransductiveGraph instead (default: False)

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from topobench.data.loaders import OGBNProductsLoader
    >>>
    >>> # Load full graph
    >>> config = OmegaConf.create({
    ...     "data_dir": "./data",
    ...     "data_name": "ogbn-products"
    ... })
    >>>
    >>> loader = OGBNProductsLoader(config)
    >>> dataset, data_dir = loader.load()
    >>> print(f"Loaded graph with {dataset[0].num_nodes} nodes")
    >>>
    >>> # Load subgraph for testing (much faster indexing!)
    >>> config.subset_nodes = 10000
    >>> loader = OGBNProductsLoader(config)
    >>> dataset, data_dir = loader.load()
    >>> print(f"Loaded subgraph with {dataset[0].num_nodes} nodes")
    >>>
    >>> # Use mock dataset (no download needed!)
    >>> config.use_mock = True
    >>> loader = OGBNProductsLoader(config)
    >>> dataset, data_dir = loader.load()
    >>> print(f"Loaded mock graph with {dataset[0].num_nodes} nodes")

    Notes
    -----
    - Requires ogb package: `pip install ogb` (unless use_mock=True)
    - First run downloads dataset (~1.5GB)
    - Subsequent runs use cached data
    - Mock dataset is perfect for testing pipelines without downloads

    References
    ----------
    OGB: Open Graph Benchmark
    https://ogb.stanford.edu/docs/nodeprop/#ogbn-products

    See Also
    --------
    topobench.data.preprocessor.ondisk_transductive.OnDiskTransductiveDataset :
        For on-disk processing of this large graph.
    topobench.data.datasets.mock_transductive_graph.MockTransductiveGraph :
        Mock dataset for testing without downloads.
    """

    def __init__(self, parameters: DictConfig) -> None:
        """Initialize OGBN-products loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration with data_dir, data_name, subset_nodes, and use_mock.
        """
        super().__init__(parameters)
        self.name = parameters.get("data_name", "ogbn-products")
        self.subset_nodes = parameters.get("subset_nodes", None)
        self.use_mock = parameters.get("use_mock", False)

    def load_dataset(self) -> tuple:
        """Load OGBN-products or mock dataset.

        Returns
        -------
        tuple
            (dataset, data_dir) where dataset is a wrapper containing
            the graph as a single PyG Data object.

        Raises
        ------
        ImportError
            If ogb package is not installed (when use_mock=False).
        RuntimeError
            If dataset download or loading fails.
        """
        # Use mock dataset if requested
        if self.use_mock:
            return self._load_mock_dataset()
        try:
            from ogb.nodeproppred import NodePropPredDataset
        except ImportError as e:
            raise ImportError(
                "ogb package is required for OGBN-products dataset. "
                "Install with: pip install ogb"
            ) from e

        try:
            # Patch torch.load for PyTorch 2.6+ compatibility
            # OGB uses pickle which requires weights_only=False
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                # Force weights_only=False for OGB compatibility
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            try:
                # Load dataset using OGB
                dataset = NodePropPredDataset(
                    name="ogbn-products", root=str(self.root_data_dir)
                )
            finally:
                # Restore original torch.load
                torch.load = original_load

            # Get graph and splits
            graph, labels = dataset[0]  # graph is a dict, labels is numpy array

            # Convert to PyG Data format
            from torch_geometric.data import Data

            # Extract edge index (already undirected in OGB)
            edge_index = torch.from_numpy(graph["edge_index"]).long()

            # Node features
            x = torch.from_numpy(graph["node_feat"]).float()

            # Labels (product categories)
            y = torch.from_numpy(labels).squeeze().long()

            # Get official splits
            split_idx = dataset.get_idx_split()
            train_idx = split_idx["train"]
            valid_idx = split_idx["valid"]
            test_idx = split_idx["test"]

            # Create masks
            num_nodes = x.shape[0]
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[train_idx] = True
            val_mask[valid_idx] = True
            test_mask[test_idx] = True

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                num_nodes=num_nodes,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
            )
            
            # Sample subgraph if requested (for faster testing)
            if self.subset_nodes is not None and self.subset_nodes < num_nodes:
                data = self._sample_subgraph(data, self.subset_nodes)

            # Wrap in a simple dataset class for compatibility
            class OGBNProductsDataset:
                """Simple wrapper for OGBN-products graph."""

                def __init__(self, data):
                    self.data = data
                    self._data_list = [data]

                def __len__(self):
                    return 1

                def __getitem__(self, idx):
                    if idx != 0:
                        raise IndexError("OGBN-products is a single graph")
                    return self.data

                @property
                def data_list(self):
                    return self._data_list

                def get_data_dir(self):
                    return str(self.root_data_dir)

            wrapped_dataset = OGBNProductsDataset(data)

            # Store root_data_dir for get_data_dir method
            wrapped_dataset.root_data_dir = self.root_data_dir

            return wrapped_dataset

        except Exception as e:
            raise RuntimeError(f"Failed to load OGBN-products dataset: {e}") from e
    
    def _load_mock_dataset(self) -> tuple:
        """Load mock transductive graph dataset.
        
        Returns
        -------
        tuple
            (dataset, data_dir) tuple with MockTransductiveGraph.
        """
        # Use subset_nodes if provided, otherwise default to 100
        # Note: Keep small! Transductive structure indexing is expensive (triangles grow quadratically)
        num_nodes = self.subset_nodes if self.subset_nodes is not None else 100
        
        dataset = MockTransductiveGraph(
            root=self.root_data_dir,
            num_nodes=num_nodes,
            avg_degree=min(10, num_nodes - 1),  # Keep degree reasonable for small graphs
            num_features=100,
            num_classes=47,
            train_ratio=0.08,
            val_ratio=0.02,
        )
        
        # Wrap in same format as real dataset
        class OGBNProductsDataset:
            """Simple wrapper for mock graph."""
            
            def __init__(self, data):
                self.data = data
                self._data_list = [data]
            
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                if idx != 0:
                    raise IndexError("Single graph dataset")
                return self.data
            
            @property
            def data_list(self):
                return self._data_list
            
            def get_data_dir(self):
                return str(self.root_data_dir)
        
        graph_data = dataset[0]
        wrapped_dataset = OGBNProductsDataset(graph_data)
        wrapped_dataset.root_data_dir = self.root_data_dir
        
        print(f"\n  🎭 Using mock transductive graph:")
        print(f"     Nodes: {graph_data.num_nodes:,}, Edges: {graph_data.edge_index.shape[1]:,}")
        print(f"     Train: {graph_data.train_mask.sum():,}, Val: {graph_data.val_mask.sum():,}, Test: {graph_data.test_mask.sum():,}")
        
        return wrapped_dataset
    
    def _sample_subgraph(self, data: Data, num_nodes: int) -> Data:
        """Sample a connected subgraph with specified number of nodes.
        
        Parameters
        ----------
        data : Data
            Full graph data.
        num_nodes : int
            Number of nodes to sample.
        
        Returns
        -------
        Data
            Subgraph with sampled nodes.
        """
        # torch and Data already imported at module level
        
        # Start from a random training node and expand via BFS
        train_nodes = torch.where(data.train_mask)[0]
        if len(train_nodes) == 0:
            # Fallback: sample any nodes
            seed_node = torch.randint(0, data.num_nodes, (1,)).item()
        else:
            seed_node = train_nodes[torch.randint(0, len(train_nodes), (1,))].item()
        
        # Expand from seed node to get connected subgraph
        subset_nodes = [seed_node]
        visited = {seed_node}
        frontier = [seed_node]
        
        # BFS to get connected nodes
        while len(subset_nodes) < num_nodes and frontier:
            current = frontier.pop(0)
            # Get neighbors
            neighbors = data.edge_index[1][data.edge_index[0] == current].tolist()
            for neighbor in neighbors:
                if neighbor not in visited and len(subset_nodes) < num_nodes:
                    visited.add(neighbor)
                    subset_nodes.append(neighbor)
                    frontier.append(neighbor)
        
        # Convert to tensor
        subset_nodes = torch.tensor(subset_nodes, dtype=torch.long)
        
        # Extract subgraph
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        node_mask[subset_nodes] = True
        
        # Get edges within subgraph
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        sub_edge_index = data.edge_index[:, edge_mask]
        
        # Reindex nodes
        node_idx = torch.full((data.num_nodes,), -1, dtype=torch.long)
        node_idx[subset_nodes] = torch.arange(len(subset_nodes))
        sub_edge_index = node_idx[sub_edge_index]
        
        # Create subgraph data
        sub_data = Data(
            x=data.x[subset_nodes],
            edge_index=sub_edge_index,
            y=data.y[subset_nodes],
            num_nodes=len(subset_nodes),
            train_mask=data.train_mask[subset_nodes],
            val_mask=data.val_mask[subset_nodes],
            test_mask=data.test_mask[subset_nodes],
        )
        
        print(f"\n  📊 Sampled subgraph: {len(subset_nodes):,} nodes, {sub_edge_index.shape[1]:,} edges")
        print(f"     Train: {sub_data.train_mask.sum():,}, Val: {sub_data.val_mask.sum():,}, Test: {sub_data.test_mask.sum():,}")
        
        return sub_data

    def load(self, **kwargs) -> tuple:
        """Load dataset and return with data directory.

        Returns
        -------
        tuple
            (dataset, data_dir) tuple.
        """
        dataset = self.load_dataset(**kwargs)
        data_dir = str(self.root_data_dir)
        return dataset, data_dir
