"""Loader for OGBN-products dataset.

This module provides a loader for the OGBN-products dataset, which is a large-scale
product co-purchasing network from Amazon with 2.4M nodes and 61M edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from omegaconf import DictConfig

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

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from topobench.data.loaders import OGBNProductsLoader
    >>>
    >>> config = OmegaConf.create({
    ...     "data_dir": "./data",
    ...     "data_name": "ogbn-products"
    ... })
    >>>
    >>> loader = OGBNProductsLoader(config)
    >>> dataset, data_dir = loader.load()
    >>> print(f"Loaded graph with {dataset[0].num_nodes} nodes")

    Notes
    -----
    - Requires ogb package: `pip install ogb`
    - First run downloads dataset (~1.5GB)
    - Subsequent runs use cached data

    References
    ----------
    OGB: Open Graph Benchmark
    https://ogb.stanford.edu/docs/nodeprop/#ogbn-products

    See Also
    --------
    topobench.data.preprocessor.ondisk_transductive.OnDiskTransductiveDataset :
        For on-disk processing of this large graph.
    """

    def __init__(self, parameters: DictConfig) -> None:
        """Initialize OGBN-products loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration with data_dir and data_name.
        """
        super().__init__(parameters)
        self.name = parameters.get("data_name", "ogbn-products")

    def load_dataset(self) -> tuple:
        """Load OGBN-products dataset.

        Returns
        -------
        tuple
            (dataset, data_dir) where dataset is a wrapper containing
            the graph as a single PyG Data object.

        Raises
        ------
        ImportError
            If ogb package is not installed.
        RuntimeError
            If dataset download or loading fails.
        """
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
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
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
