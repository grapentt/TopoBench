"""Streaming structure detection for transductive learning.

This module provides memory-efficient algorithms for detecting topological
structures (cliques, cycles, etc.) in large graphs without loading all
structures into memory at once.
"""

import networkx as nx
from typing import Iterator


def enumerate_cliques_streaming(
    graph: nx.Graph, max_size: int = None
) -> Iterator[tuple[int, list[int]]]:
    """Enumerate all maximal cliques in a graph using streaming approach.

    This function yields cliques one at a time, ensuring constant memory
    usage regardless of the total number of structures found. This is
    critical for large graphs where the number of cliques can be enormous.

    Parameters
    ----------
    graph : nx.Graph
        Input graph to find cliques in.
    max_size : int, optional
        Maximum clique size to return. If None, returns all maximal cliques
        (default: None).

    Yields
    ------
    tuple of (int, list of int)
        Tuple of (structure_id, sorted_node_list) for each clique.
        structure_id is assigned sequentially starting from 0.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> cliques = list(enumerate_cliques_streaming(G, max_size=3))
    >>> print(f"Found {len(cliques)} triangles")

    Notes
    -----
    - Uses NetworkX's find_cliques() which is memory-efficient
    - Cliques are yielded in arbitrary order (not sorted by size)
    - For simplicial complexes of dimension d, use max_size=d+1
    - Memory usage: O(k) where k is the size of largest clique

    See Also
    --------
    enumerate_k_cliques_streaming : Find cliques of specific size only.
    """
    structure_id = 0

    # NetworkX find_cliques yields maximal cliques one at a time
    for clique in nx.find_cliques(graph):
        # Filter by size if requested
        if max_size is None or len(clique) <= max_size:
            yield (structure_id, sorted(clique))
            structure_id += 1


def enumerate_k_cliques_streaming(
    graph: nx.Graph, k: int
) -> Iterator[tuple[int, list[int]]]:
    """Enumerate all k-cliques in a graph using streaming approach.

    This function finds all cliques of exactly size k (k-cliques), which
    correspond to (k-1)-simplices in simplicial complex lifting. For example,
    k=3 finds all triangles (2-simplices).

    Parameters
    ----------
    graph : nx.Graph
        Input graph to find k-cliques in.
    k : int
        Size of cliques to find. Must be >= 2.

    Yields
    ------
    tuple of (int, list of int)
        Tuple of (structure_id, sorted_node_list) for each k-clique.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.complete_graph(5)
    >>> triangles = list(enumerate_k_cliques_streaming(G, k=3))
    >>> print(f"Complete graph K5 has {len(triangles)} triangles")

    Notes
    -----
    - OPTIMIZED: For k=3 (triangles), uses direct enumeration algorithm
    - For k>3, falls back to NetworkX's enumerate_all_cliques()
    - Memory usage: O(k) constant per clique
    - Performance: O(n * d^2) for triangles where n=nodes, d=avg degree

    See Also
    --------
    enumerate_cliques_streaming : Find all maximal cliques.
    enumerate_triangles_streaming : Optimized triangle enumeration.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    # OPTIMIZATION: For triangles (most common case), use direct algorithm
    if k == 3:
        structure_id = 0
        
        # Direct triangle enumeration: O(n * d^2) instead of exponential
        # Iterate through each node and check neighbor pairs
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            
            # Check all pairs of neighbors
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    # If neighbors are connected, we have a triangle
                    if graph.has_edge(n1, n2):
                        # Ensure canonical ordering (avoid duplicates)
                        triangle = sorted([node, n1, n2])
                        if triangle[0] == node:  # Only yield if node is smallest
                            yield (structure_id, triangle)
                            structure_id += 1
    else:
        # For k > 3, use NetworkX's enumerate_all_cliques
        # (less common case, generic algorithm acceptable)
        structure_id = 0
        
        for clique in nx.enumerate_all_cliques(graph):
            if len(clique) == k:
                yield (structure_id, sorted(clique))
                structure_id += 1
            elif len(clique) > k:
                # enumerate_all_cliques yields in ascending size order
                # Once we exceed k, we can stop
                break


def enumerate_triangles_streaming(
    graph: nx.Graph,
) -> Iterator[tuple[int, list[int]]]:
    """Enumerate all triangles (3-cliques) in a graph efficiently.

    This is a specialized, optimized version for finding triangles, which
    are the most common structure in simplicial complex lifting.

    Parameters
    ----------
    graph : nx.Graph
        Input graph to find triangles in.

    Yields
    ------
    tuple of (int, list of int)
        Tuple of (structure_id, sorted_node_list) for each triangle.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> triangles = list(enumerate_triangles_streaming(G))
    >>> print(f"Karate Club has {len(triangles)} triangles")

    Notes
    -----
    - This is equivalent to enumerate_k_cliques_streaming(graph, k=3)
    - Provided as convenience function for common use case
    - NetworkX has optimized triangle finding algorithms

    See Also
    --------
    enumerate_k_cliques_streaming : General k-clique enumeration.
    """
    return enumerate_k_cliques_streaming(graph, k=3)


def count_triangles(graph: nx.Graph) -> int:
    """Count total number of triangles in graph.

    This is faster than enumerating when only the count is needed.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.

    Returns
    -------
    int
        Total number of triangles in the graph.

    Notes
    -----
    Uses NetworkX's optimized triangle counting algorithm.
    """
    # Sum of triangles per node divided by 3 (each triangle counted 3 times)
    triangles_dict = nx.triangles(graph)
    return sum(triangles_dict.values()) // 3


def build_clique_index(
    graph: nx.Graph, index_backend, max_size: int = None, show_progress: bool = True
) -> None:
    """Build structure index for all cliques in graph.

    This function detects all cliques and stores them in the provided
    index backend, enabling efficient queries during training.

    Parameters
    ----------
    graph : nx.Graph
        Input graph to analyze.
    index_backend : AbstractIndexBackend
        Opened index backend to store structures in.
    max_size : int, optional
        Maximum clique size to index. If None, indexes all maximal cliques.
        If specified, indexes all k-cliques for k <= max_size (default: None).
    show_progress : bool, optional
        If True, display progress bar during indexing (default: True).

    Examples
    --------
    >>> from topobench.data.index import SQLiteIndexBackend
    >>> import networkx as nx
    >>>
    >>> G = nx.karate_club_graph()
    >>> backend = SQLiteIndexBackend(data_dir="/tmp/index")
    >>> backend.open()
    >>> build_clique_index(G, backend, max_size=3)
    >>> print(f"Indexed {backend.count_structures()} triangles")
    >>> backend.close()

    Notes
    -----
    - Uses streaming enumeration for constant memory usage
    - Progress bar displays structures processed (requires tqdm)
    - Index must be opened before calling this function
    - When max_size is specified, enumerates all cliques of that size
      (not just maximal cliques filtered by size)
    """
    # If max_size specified, enumerate k-cliques of that exact size
    # (most common case: triangles for simplicial complexes)
    if max_size is not None:
        clique_iterator = enumerate_k_cliques_streaming(graph, k=max_size)
    else:
        # Enumerate all maximal cliques
        clique_iterator = enumerate_cliques_streaming(graph, max_size=None)
    
    # Wrap with progress bar if requested
    if show_progress:
        try:
            from tqdm import tqdm
            # We don't know total count in advance, so use unbounded progress bar
            clique_iterator = tqdm(
                clique_iterator,
                desc=f"Indexing {max_size}-cliques" if max_size else "Indexing cliques",
                unit=" structures",
                mininterval=0.5,
            )
        except ImportError:
            # tqdm not available, proceed without progress bar
            pass
    
    index_backend.insert_batch(clique_iterator)


def build_triangle_index(graph: nx.Graph, index_backend) -> None:
    """Build structure index for all triangles in graph.

    Specialized version of build_clique_index for triangles only.

    Parameters
    ----------
    graph : nx.Graph
        Input graph to analyze.
    index_backend : AbstractIndexBackend
        Opened index backend to store structures in.

    Examples
    --------
    >>> from topobench.data.index import SQLiteIndexBackend
    >>> import networkx as nx
    >>>
    >>> G = nx.karate_club_graph()
    >>> backend = SQLiteIndexBackend(data_dir="/tmp/index")
    >>> backend.open()
    >>> build_triangle_index(G, backend)
    >>> backend.close()

    Notes
    -----
    Equivalent to build_clique_index(graph, index_backend, max_size=3).
    """
    triangle_iterator = enumerate_triangles_streaming(graph)
    index_backend.insert_batch(triangle_iterator)
