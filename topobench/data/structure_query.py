"""High-level structure query engine for transductive learning.

This module provides a user-friendly interface for querying topological
structures indexed for transductive learning, with correctness guarantees.
"""

from pathlib import Path
from typing import Any

import networkx as nx

from topobench.data.index import SQLiteIndexBackend
from topobench.data.structure_detection import build_clique_index


class StructureQueryEngine:
    """High-level query interface for indexed topological structures.

    This class provides a simple API for querying structures that have been
    indexed for a transductive learning graph. It handles backend initialization,
    structure detection, and provides correctness guarantees.

    Parameters
    ----------
    graph : nx.Graph
        The graph to query structures from.
    index_dir : str or Path
        Directory for storing the structure index.
    max_structure_size : int, optional
        Maximum size of structures to index (e.g., 3 for triangles).
        If None, indexes all maximal cliques (default: None).
    backend : str, optional
        Backend type to use ('sqlite' or 'auto'). Default: 'auto'.
    force_rebuild : bool, optional
        If True, rebuild index even if it exists (default: False).

    Attributes
    ----------
    graph : nx.Graph
        The input graph.
    backend : AbstractIndexBackend
        The storage backend for structure queries.
    num_structures : int
        Total number of structures indexed.

    Examples
    --------
    >>> import networkx as nx
    >>> from topobench.data.structure_query import StructureQueryEngine
    >>>
    >>> # Create query engine for Karate Club graph
    >>> G = nx.karate_club_graph()
    >>> engine = StructureQueryEngine(G, index_dir="/tmp/karate_index", max_structure_size=3)
    >>> engine.build_index()
    >>>
    >>> # Query structures for a batch of nodes
    >>> batch_nodes = [0, 1, 2, 3, 4]
    >>> structures = engine.query_batch(batch_nodes)
    >>> print(f"Found {len(structures)} structures in batch")
    >>>
    >>> engine.close()

    Notes
    -----
    - The index is built once and persisted to disk
    - Subsequent queries reuse the index for fast lookups
    - Always call close() or use context manager to ensure cleanup

    See Also
    --------
    topobench.data.index.SQLiteIndexBackend :
        Underlying storage backend.
    topobench.data.structure_detection.build_clique_index :
        Structure detection algorithm.
    """

    def __init__(
        self,
        graph: nx.Graph,
        index_dir: str | Path,
        max_structure_size: int | None = None,
        backend: str = "auto",
        force_rebuild: bool = False,
    ) -> None:
        """Initialize structure query engine.

        Parameters
        ----------
        graph : nx.Graph
            The graph to query structures from.
        index_dir : str or Path
            Directory for storing the structure index.
        max_structure_size : int, optional
            Maximum size of structures to index (default: None for all maximal).
        backend : str, optional
            Backend type ('sqlite' or 'auto', default: 'auto').
        force_rebuild : bool, optional
            Rebuild index even if exists (default: False).
        """
        self.graph = graph
        self.index_dir = Path(index_dir)
        self.max_structure_size = max_structure_size
        self.force_rebuild = force_rebuild

        # Initialize backend
        if backend == "auto" or backend == "sqlite":
            self.backend = SQLiteIndexBackend(data_dir=str(self.index_dir))
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.num_structures = 0
        self._is_open = False

    def open(self) -> None:
        """Open connection to index backend.

        Raises
        ------
        RuntimeError
            If opening backend fails.
        """
        self.backend.open()
        self._is_open = True

    def close(self) -> None:
        """Close connection to index backend."""
        if self._is_open:
            self.backend.close()
            self._is_open = False

    def build_index(self) -> None:
        """Build or load structure index.

        This method checks if an index exists. If not (or if force_rebuild=True),
        it builds the index by detecting all structures and storing them.

        Notes
        -----
        This can take time for large graphs. Progress is displayed for large graphs.
        """
        if not self._is_open:
            raise RuntimeError("Backend not open. Call open() first.")

        # Check if index exists and we're not forcing rebuild
        if not self.force_rebuild and self.backend.exists():
            self.num_structures = self.backend.count_structures()
            print(f"✓ Loaded existing index: {self.num_structures} structures")
            return

        # Clear if force rebuild
        if self.force_rebuild and self.backend.exists():
            self.backend.clear()

        # Build index
        print(f"Building index for graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        build_clique_index(
            self.graph, self.backend, max_size=self.max_structure_size
        )

        self.num_structures = self.backend.count_structures()
        print(f"✓ Indexed {self.num_structures} structures")

    def query_batch(
        self, node_ids: list[int], fully_contained: bool = True
    ) -> list[tuple[int, list[int]]]:
        """Query structures relevant to a batch of nodes.

        Parameters
        ----------
        node_ids : list of int
            Node IDs in the batch to query for.
        fully_contained : bool, optional
            If True, return only structures where ALL nodes are in node_ids.
            If False, return structures where ANY node is in node_ids
            (default: True).

        Returns
        -------
        list of (int, list of int)
            List of (structure_id, nodes) tuples for structures matching the query.

        Notes
        -----
        For transductive learning, fully_contained=True is typically required
        to ensure structures don't reference nodes outside the batch.
        """
        if not self._is_open:
            raise RuntimeError("Backend not open. Call open() first.")

        return self.backend.query_by_nodes(
            node_ids, fully_contained=fully_contained
        )

    def query_node_structures(
        self, node_id: int
    ) -> list[tuple[int, list[int]]]:
        """Query all structures containing a specific node.

        Parameters
        ----------
        node_id : int
            Node ID to query structures for.

        Returns
        -------
        list of (int, list of int)
            List of (structure_id, nodes) tuples containing the node.
        """
        return self.query_batch([node_id], fully_contained=False)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the indexed structures.

        Returns
        -------
        dict
            Dictionary with statistics:
            - num_structures: Total structures indexed
            - num_nodes: Number of nodes in graph
            - num_edges: Number of edges in graph
            - index_dir: Path to index directory
        """
        return {
            "num_structures": self.num_structures,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "index_dir": str(self.index_dir),
            "max_structure_size": self.max_structure_size,
        }

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StructureQueryEngine("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"structures={self.num_structures})"
        )


def verify_query_correctness(
    query_engine: StructureQueryEngine, batch_nodes: list[int]
) -> dict[str, Any]:
    """Verify query correctness against in-memory baseline.

    This function validates that the query engine returns the same structures
    as a naive in-memory enumeration, ensuring 100% correctness.

    Parameters
    ----------
    query_engine : StructureQueryEngine
        Query engine to validate.
    batch_nodes : list of int
        Batch of nodes to query for.

    Returns
    -------
    dict
        Validation results with keys:
        - correct: bool (True if results match)
        - engine_count: int (structures from query engine)
        - baseline_count: int (structures from baseline)
        - missing: list (structures in baseline but not in engine)
        - extra: list (structures in engine but not in baseline)

    Notes
    -----
    This is intended for validation/testing, not production use.
    """
    from topobench.data.structure_detection import (
        enumerate_cliques_streaming,
        enumerate_k_cliques_streaming,
    )

    # Get results from query engine
    engine_results = query_engine.query_batch(batch_nodes, fully_contained=True)
    engine_structures = {tuple(sorted(nodes)) for _, nodes in engine_results}

    # Get baseline: enumerate all cliques and filter
    # Use same logic as build_clique_index
    batch_set = set(batch_nodes)
    baseline_structures = set()

    if query_engine.max_structure_size is not None:
        # Enumerate k-cliques of specific size
        clique_iter = enumerate_k_cliques_streaming(
            query_engine.graph, k=query_engine.max_structure_size
        )
    else:
        # Enumerate all maximal cliques
        clique_iter = enumerate_cliques_streaming(
            query_engine.graph, max_size=None
        )

    for _, nodes in clique_iter:
        # Check if fully contained in batch
        if set(nodes).issubset(batch_set):
            baseline_structures.add(tuple(sorted(nodes)))

    # Compare
    missing = baseline_structures - engine_structures
    extra = engine_structures - baseline_structures

    return {
        "correct": len(missing) == 0 and len(extra) == 0,
        "engine_count": len(engine_structures),
        "baseline_count": len(baseline_structures),
        "missing": list(missing),
        "extra": list(extra),
    }
