"""On-disk transductive dataset for large-scale graph learning.

This module provides memory-efficient transductive learning on large graphs
by indexing topological structures offline and querying them during training.
"""

from pathlib import Path
from typing import Any

import networkx as nx
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from topobench.data.structure_query import StructureQueryEngine


class OnDiskTransductivePreprocessor(torch.utils.data.Dataset):
    """On-disk preprocessor for transductive learning on large graphs.

    This preprocessor provides efficient structure detection, indexing, and on-demand
    querying for large-scale transductive graph learning without loading all
    structures into memory. It builds an index of topological structures (e.g.,
    triangles, cliques) that can be queried during mini-batch training.

    Parameters
    ----------
    graph_data : Data
        PyTorch Geometric Data object containing the graph.
    data_dir : str or Path
        Directory for storing the structure index.
    transforms_config : DictConfig, optional
        Configuration for topological transforms (default: None).
    max_structure_size : int, optional
        Maximum size of structures to index (e.g., 3 for triangles).
        If None, indexes all maximal cliques (default: 3).
    force_rebuild : bool, optional
        If True, rebuild index even if it exists (default: False).
    **kwargs : dict
        Additional arguments.

    Attributes
    ----------
    graph_data : Data
        The input graph data.
    query_engine : StructureQueryEngine
        Query engine for structure lookups.
    num_nodes : int
        Number of nodes in the graph.
    num_structures : int
        Total number of indexed structures.

    Examples
    --------
    >>> from torch_geometric.datasets import Planetoid
    >>> from topobench.data.preprocessor import OnDiskTransductiveDataset
    >>>
    >>> # Load Cora dataset
    >>> dataset = Planetoid(root='/tmp/Cora', name='Cora')
    >>> data = dataset[0]
    >>>
    >>> # Create transductive dataset
    >>> trans_dataset = OnDiskTransductiveDataset(
    ...     graph_data=data,
    ...     data_dir="/tmp/cora_index",
    ...     max_structure_size=3
    ... )
    >>>
    >>> # Build index (one-time operation)
    >>> trans_dataset.build_index()
    >>>
    >>> # Query structures for a batch of nodes
    >>> batch_nodes = [0, 1, 2, 3, 4]
    >>> structures = trans_dataset.query_batch(batch_nodes)
    >>> print(f"Found {len(structures)} structures in batch")

    Notes
    -----
    - The index is built once and persisted to disk
    - Subsequent runs reuse the index for fast startup
    - Memory usage is constant O(1) regardless of graph size
    - Compatible with mini-batch training via node sampling

    See Also
    --------
    topobench.data.structure_query.StructureQueryEngine :
        Underlying query engine for structure lookups.
    topobench.data.preprocessor.ondisk_inductive.OnDiskInductiveDataset :
        Inductive learning variant.
    """

    def __init__(
        self,
        graph_data: Data,
        data_dir: str | Path,
        transforms_config: DictConfig | None = None,
        max_structure_size: int | None = 3,
        force_rebuild: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize OnDiskTransductiveDataset.

        Parameters
        ----------
        graph_data : Data
            PyTorch Geometric Data object containing the graph.
        data_dir : str or Path
            Directory for storing the structure index.
        transforms_config : DictConfig, optional
            Configuration for transforms (default: None).
        max_structure_size : int, optional
            Maximum structure size to index (default: 3 for triangles).
        force_rebuild : bool, optional
            Rebuild index if exists (default: False).
        **kwargs : dict
            Additional arguments.
        """
        super().__init__()
        self.graph_data = graph_data
        self.data_dir = Path(data_dir)
        self.transforms_config = transforms_config
        self.max_structure_size = max_structure_size
        self.force_rebuild = force_rebuild

        # Note: transforms_config is stored for compatibility with TopoBench framework
        # and will be used by OnDiskTransductiveCollate during batch construction.
        # The transductive preprocessor indexes raw structures, and transforms are
        # applied during collation (unlike inductive, where transforms are applied
        # during preprocessing).

        # Convert PyG Data to NetworkX for structure detection
        self.nx_graph = self._pyg_to_networkx(graph_data)

        # Initialize query engine
        self.query_engine = StructureQueryEngine(
            graph=self.nx_graph,
            index_dir=self.data_dir,
            max_structure_size=self.max_structure_size,
            force_rebuild=self.force_rebuild,
        )

        self.num_nodes = graph_data.num_nodes
        self.num_structures = 0
        self._index_built = False

    def _pyg_to_networkx(self, data: Data) -> nx.Graph:
        """Convert PyTorch Geometric Data to NetworkX Graph.

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object.

        Returns
        -------
        nx.Graph
            NetworkX graph.
        """
        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(range(data.num_nodes))

        # Add edges
        edge_index = data.edge_index.cpu().numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)

        return G

    def build_index(self) -> None:
        """Build or load structure index.

        This method detects all topological structures in the graph and
        indexes them for efficient batch queries during training.

        Notes
        -----
        This is a one-time operation. The index is persisted to disk and
        reused in subsequent runs unless force_rebuild=True.
        """
        print(f"Building index for graph: {self.num_nodes} nodes, {self.nx_graph.number_of_edges()} edges")

        self.query_engine.open()
        self.query_engine.build_index()
        self.num_structures = self.query_engine.num_structures
        self._index_built = True

        print(f"✓ Index ready: {self.num_structures} structures indexed")

    def query_batch(
        self, node_ids: list[int], fully_contained: bool = True
    ) -> list[tuple[int, list[int]]]:
        """Query structures for a batch of nodes.

        Parameters
        ----------
        node_ids : list of int
            Node IDs in the batch.
        fully_contained : bool, optional
            If True, return only structures where ALL nodes are in the batch
            (default: True for transductive correctness).

        Returns
        -------
        list of (int, list of int)
            List of (structure_id, nodes) tuples for structures in the batch.

        Notes
        -----
        For transductive learning, fully_contained=True ensures that
        structures don't reference nodes outside the training batch.
        """
        if not self._index_built:
            raise RuntimeError(
                "Index not built. Call build_index() first."
            )

        return self.query_engine.query_batch(
            node_ids, fully_contained=fully_contained
        )

    def get_subgraph(self, node_ids: list[int]) -> Data:
        """Extract subgraph for a batch of nodes with their structures.

        Parameters
        ----------
        node_ids : list of int
            Node IDs to include in subgraph.

        Returns
        -------
        Data
            PyTorch Geometric Data object for the subgraph.

        Notes
        -----
        This creates a subgraph containing:
        - The specified nodes and their features
        - Edges between these nodes
        - Topological structures (fully contained in the batch)
        """
        # Get structures for this batch
        structures = self.query_batch(node_ids, fully_contained=True)

        # Extract subgraph from original data
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        node_mask[node_ids] = True

        # Get edge mask
        edge_index = self.graph_data.edge_index
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]

        # Create subgraph data
        subgraph_data = Data()

        # Node features
        if hasattr(self.graph_data, 'x') and self.graph_data.x is not None:
            subgraph_data.x = self.graph_data.x[node_mask]

        # Labels (if present)
        if hasattr(self.graph_data, 'y') and self.graph_data.y is not None:
            subgraph_data.y = self.graph_data.y[node_mask]

        # Edges (reindex to subgraph)
        subgraph_edge_index = edge_index[:, edge_mask]

        # Reindex nodes
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
        reindexed_edges = torch.tensor([
            [node_mapping[src.item()], node_mapping[dst.item()]]
            for src, dst in subgraph_edge_index.t()
        ], dtype=torch.long).t()

        subgraph_data.edge_index = reindexed_edges

        # Add structure information
        subgraph_data.structures = structures
        subgraph_data.num_structures = len(structures)

        return subgraph_data

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics.

        Returns
        -------
        dict
            Statistics including num_nodes, num_edges, num_structures.
        """
        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": self.nx_graph.number_of_edges(),
            "num_structures": self.num_structures,
            "max_structure_size": self.max_structure_size,
            "index_dir": str(self.data_dir),
        }
        return stats

    def close(self) -> None:
        """Close query engine and cleanup resources."""
        if self._index_built:
            self.query_engine.close()
            self._index_built = False

    def __len__(self) -> int:
        """Return number of nodes in the graph.

        Returns
        -------
        int
            Number of nodes.

        Notes
        -----
        For transductive learning, this typically represents the number
        of training nodes, not the total graph size.
        """
        return self.num_nodes

    def __getitem__(self, idx: int | list[int]) -> Data:
        """Get subgraph for node(s).

        Parameters
        ----------
        idx : int or list of int
            Node ID or list of node IDs.

        Returns
        -------
        Data
            Subgraph data for the specified node(s).

        Notes
        -----
        This method supports both single-node and batch queries.
        """
        if isinstance(idx, int):
            idx = [idx]

        return self.get_subgraph(idx)

    def __repr__(self) -> str:
        """String representation.

        Returns
        -------
        str
            String representation of dataset.
        """
        return (
            f"OnDiskTransductiveDataset("
            f"num_nodes={self.num_nodes}, "
            f"num_structures={self.num_structures})"
        )

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass
