"""Synthetic large transductive dataset for validation testing."""

import os.path as osp
from typing import ClassVar

import networkx as nx
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


class SyntheticLargeTransductiveDataset(InMemoryDataset):
    """Synthetic large single graph for transductive learning.

    This dataset generates a single large synthetic graph with controlled
    parameters to demonstrate memory limitations of in-memory preprocessing.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters:
        - num_nodes : int
            Number of nodes in the graph
        - degree : int
            Average degree per node
        - num_features : int
            Number of node features
        - num_classes : int
            Number of classes for node classification
    """

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters

        # Dataset parameters
        self._num_nodes = parameters.get("num_nodes", 15000)
        self._degree = parameters.get("degree", 60)
        self._num_node_features = parameters.get("num_features", 16)
        self._num_classes = parameters.get("num_classes", 10)

        super().__init__(root)

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return (
            f"{self.name}(root={self.root}, num_nodes={self._num_nodes}, "
            f"degree={self._degree})"
        )

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        config_str = f"n{self._num_nodes}_d{self._degree}"
        self.processed_root = osp.join(self.root, self.name, config_str)
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names (none needed for synthetic).

        Returns
        -------
        list[str]
            Empty list (no raw files needed).
        """
        return []

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        """Download is not needed for synthetic data."""
        pass

    def process(self) -> None:
        """Generate synthetic large graph and save it.

        This method creates a single large synthetic graph using
        Watts-Strogatz model for transductive node classification.
        """
        print(f"Generating synthetic transductive graph...")
        print(f"  Nodes: {self._num_nodes}")
        print(f"  Degree: {self._degree}")
        print(f"  Features: {self._num_node_features}")
        print(f"  Classes: {self._num_classes}")
        print()

        # Generate Watts-Strogatz graph (creates many triangles)
        G = nx.watts_strogatz_graph(
            n=self._num_nodes,
            k=self._degree,
            p=0.5,  # Rewiring probability
            seed=42,
        )

        # Convert to PyG Data
        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        # Random features
        x = torch.randn(self._num_nodes, self._num_node_features)

        # Random node labels
        y = torch.randint(0, self._num_classes, (self._num_nodes,))

        # Transductive masks
        n = self._num_nodes
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        train_mask[: int(0.6 * n)] = True
        val_mask[int(0.6 * n) : int(0.8 * n)] = True
        test_mask[int(0.8 * n) :] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=n,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        print(f"✓ Generated graph:")
        print(f"  Nodes: {n}")
        print(f"  Edges: {G.number_of_edges():,}")
        print(f"  Train nodes: {train_mask.sum()}")
        print(f"  Val nodes: {val_mask.sum()}")
        print(f"  Test nodes: {test_mask.sum()}")
        print()

        # For transductive, we wrap single graph in list
        data_list = [data]

        # Collate the graph
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache

        # Save processed data
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

        print(f"✓ Dataset saved to {self.processed_paths[0]}")
