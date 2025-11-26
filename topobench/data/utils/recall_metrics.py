"""Recall metrics for evaluating structure completeness in transductive learning.

This module provides functions to compute and track topological structure
recall over training epochs, enabling comparison between complete indexing
approaches and progressive recovery methods.
"""

import torch
from torch_geometric.data import Data


def compute_structure_recall(
    candidate_structures: set,
    golden_structures: set,
    match_type: str = "strict",
) -> float:
    """Compute recall: |candidate ∩ golden| / |golden|.

    Measures the fraction of golden structures that are present in the
    candidate set. This quantifies completeness of structure recovery.

    Parameters
    ----------
    candidate_structures : Set
        Structures found in batches (e.g., from mini-batch sampling).
    golden_structures : Set
        All structures in the full graph (ground truth).
    match_type : str, optional
        Matching criterion:
        - "strict": Exact match (frozenset equality)
        - "partial": Partial overlap (subset match)
        Default: "strict"

    Returns
    -------
    float
        Recall score in [0.0, 1.0], where 1.0 means 100% complete.

    Examples
    --------
    >>> golden = {frozenset([0, 1, 2]), frozenset([1, 2, 3])}
    >>> candidate = {frozenset([0, 1, 2])}
    >>> recall = compute_structure_recall(candidate, golden)
    >>> print(f"Recall: {recall:.1%}")  # 50%

    Notes
    -----
    - Strict matching is recommended for transductive learning
    - Partial matching can be useful for approximate methods
    """
    if not golden_structures:
        return 1.0  # Vacuously true

    if match_type == "strict":
        intersection = candidate_structures & golden_structures
    elif match_type == "partial":
        # Count partial overlaps
        intersection = {
            c
            for c in candidate_structures
            if any(c.issubset(g) or g.issubset(c) for g in golden_structures)
        }
    else:
        raise ValueError(f"Unknown match_type: {match_type}")

    return len(intersection) / len(golden_structures)


def compute_cumulative_recall(
    epoch_structures: list[set], golden_structures: set
) -> list[float]:
    """Track cumulative structure discovery over epochs.

    This function simulates progressive recovery approaches by tracking
    how many structures are discovered cumulatively across epochs.

    Parameters
    ----------
    epoch_structures : List[Set]
        List of structure sets, one per epoch.
    golden_structures : Set
        All structures in the full graph.

    Returns
    -------
    List[float]
        List of cumulative recall scores, one per epoch.

    Examples
    --------
    >>> golden = {frozenset([0, 1, 2]), frozenset([1, 2, 3]), frozenset([2, 3, 4])}
    >>> epoch1 = {frozenset([0, 1, 2])}
    >>> epoch2 = {frozenset([1, 2, 3])}
    >>> epoch3 = {frozenset([2, 3, 4])}
    >>> recalls = compute_cumulative_recall([epoch1, epoch2, epoch3], golden)
    >>> # [0.33, 0.67, 1.0] - progressive recovery!

    Notes
    -----
    - Useful for comparing complete indexing vs progressive approaches
    - Complete indexing should show 1.0 from epoch 1
    - Progressive recovery gradually increases over epochs
    """
    cumulative = set()
    recalls = []

    for epoch_set in epoch_structures:
        cumulative.update(epoch_set)
        recall = len(cumulative & golden_structures) / len(golden_structures)
        recalls.append(recall)

    return recalls


def extract_structures_from_data(
    data: Data, structure_type: str = "triangles"
) -> set[frozenset]:
    """Extract topological structures from PyG Data object.

    Parses incidence matrices or edge lists to extract structures
    as sets of node IDs.

    Parameters
    ----------
    data : Data
        PyTorch Geometric Data object with incidence matrices.
    structure_type : str, optional
        Type of structure to extract:
        - "triangles": 2-simplices (from incidence_2)
        - "edges": 1-simplices (from edge_index)
        - "cliques": General cliques (from structures attribute)
        Default: "triangles"

    Returns
    -------
    Set[frozenset]
        Set of structures, each represented as a frozenset of node IDs.

    Examples
    --------
    >>> # Extract triangles from a batch
    >>> structures = extract_structures_from_data(batch_data, "triangles")
    >>> print(f"Found {len(structures)} triangles")

    Notes
    -----
    - Structures are represented as frozensets for hashability
    - Node IDs must be in the local (batch) indexing space
    - Compatible with TopoBench transform outputs
    """
    structures = set()

    if structure_type == "triangles":
        # Extract from incidence_2 (node-to-triangle incidence)
        if hasattr(data, "incidence_2") and data.incidence_2 is not None:
            incidence = data.incidence_2
            # incidence_2 is [num_nodes x num_triangles] sparse matrix
            # Each column represents a triangle
            if hasattr(incidence, "coalesce"):
                incidence = incidence.coalesce()

            # Get indices: [2, num_entries] where row 0 is node, row 1 is triangle
            indices = incidence.indices()

            # Group by triangle ID
            triangle_dict: dict[int, list[int]] = {}
            for i in range(indices.shape[1]):
                node_id = indices[0, i].item()
                triangle_id = indices[1, i].item()

                if triangle_id not in triangle_dict:
                    triangle_dict[triangle_id] = []
                triangle_dict[triangle_id].append(node_id)

            # Convert to frozensets
            for nodes in triangle_dict.values():
                if len(nodes) == 3:  # Valid triangle
                    structures.add(frozenset(nodes))

    elif structure_type == "edges":
        # Extract edges
        if hasattr(data, "edge_index") and data.edge_index is not None:
            edge_index = data.edge_index
            # Create frozensets of size 2 (undirected edges)
            seen = set()
            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i].item())
                dst = int(edge_index[1, i].item())
                edge = frozenset([src, dst])
                if edge not in seen:
                    structures.add(edge)
                    seen.add(edge)

    elif structure_type == "cliques":
        # Extract from 'structures' attribute (if available)
        # Format: list of (structure_id, nodes) tuples
        if hasattr(data, "structures") and data.structures is not None:
            for struct_id, nodes in data.structures:
                structures.add(frozenset(nodes))
        # Also check if structures are in incidence_2 (triangles)
        elif hasattr(data, "incidence_2") and data.incidence_2 is not None:
            # Fall back to triangle extraction
            return extract_structures_from_data(data, "triangles")

    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")

    return structures


def extract_structures_from_raw_data(
    data: Data, max_clique_size: int = 3
) -> set[frozenset]:
    """Extract structures directly from graph topology.

    This function detects structures by analyzing the edge_index,
    useful for creating golden reference sets.

    Parameters
    ----------
    data : Data
        PyTorch Geometric Data with edge_index.
    max_clique_size : int, optional
        Maximum clique size to detect (default: 3 for triangles).

    Returns
    -------
    Set[frozenset]
        Set of detected structures.

    Notes
    -----
    - This is slower than parsing incidence matrices
    - Used for creating ground truth reference
    - Uses NetworkX for clique detection
    """
    import networkx as nx

    # Convert to NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))

    edge_index = data.edge_index.cpu()
    edges = [
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.shape[1])
    ]
    G.add_edges_from(edges)

    structures = set()

    if max_clique_size == 2:
        # Just extract edges
        for u, v in G.edges():
            structures.add(frozenset([u, v]))
    elif max_clique_size == 3:
        # Extract triangles
        triangles = [
            frozenset(tri)
            for tri in nx.enumerate_all_cliques(G)
            if len(tri) == 3
        ]
        structures.update(triangles)
    else:
        # Extract all cliques up to max size
        for clique in nx.enumerate_all_cliques(G):
            if len(clique) <= max_clique_size:
                structures.add(frozenset(clique))

    return structures


def compute_edge_density(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Compute edge density of a subgraph.

    Edge density is the ratio of actual edges to maximum possible edges
    in an undirected graph: density = 2*E / (N*(N-1))

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index tensor [2, num_edges].
    num_nodes : int
        Number of nodes in the subgraph.

    Returns
    -------
    float
        Edge density in [0.0, 1.0], where 1.0 is a complete graph.

    Examples
    --------
    >>> # Complete graph: 3 nodes, 3 edges (undirected)
    >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    >>> density = compute_edge_density(edge_index, num_nodes=3)
    >>> print(f"Density: {density:.2f}")  # 1.0

    Notes
    -----
    - For undirected graphs, edge_index typically has both directions
    - We divide by 2 to account for bidirectional edges
    - Higher density = better community preservation
    """
    if num_nodes <= 1:
        return 0.0

    # Count unique edges (divide by 2 for undirected)
    num_edges = edge_index.shape[1] // 2

    # Maximum edges in undirected graph: n*(n-1)/2
    max_edges = num_nodes * (num_nodes - 1) // 2

    return num_edges / max_edges if max_edges > 0 else 0.0


def compute_intra_cluster_ratio(
    edge_index: torch.Tensor, cluster_labels: torch.Tensor
) -> float:
    """Compute fraction of edges within the same cluster.

    Measures how well community structure is preserved. Higher values
    indicate better cluster cohesion.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index tensor [2, num_edges].
    cluster_labels : torch.Tensor
        Cluster assignment for each node [num_nodes].

    Returns
    -------
    float
        Ratio of intra-cluster edges in [0.0, 1.0].

    Examples
    --------
    >>> # 4 nodes in 2 clusters, 3 intra-cluster edges, 1 inter-cluster edge
    >>> edge_index = torch.tensor([[0, 1, 2, 2], [1, 0, 3, 1]])
    >>> labels = torch.tensor([0, 0, 1, 1])
    >>> ratio = compute_intra_cluster_ratio(edge_index, labels)
    >>> # 3/4 = 0.75

    Notes
    -----
    - Higher ratio = better community preservation
    - Cluster-aware sampling should have higher ratios
    - Random sampling typically has lower ratios
    """
    if edge_index.shape[1] == 0:
        return 1.0  # No edges, vacuously true

    intra_count = 0
    total_count = edge_index.shape[1]

    for i in range(total_count):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()

        if cluster_labels[src] == cluster_labels[dst]:
            intra_count += 1

    return intra_count / total_count


def summarize_batch_metrics(
    batch_data: Data,
    golden_structures: set[frozenset],
    structure_type: str = "triangles",
) -> dict[str, float]:
    """Compute comprehensive metrics for a batch.

    Combines multiple metrics into a single summary dictionary.

    Parameters
    ----------
    batch_data : Data
        Batch data with structures and topology.
    golden_structures : Set[frozenset]
        Golden reference structures.
    structure_type : str, optional
        Type of structure to evaluate (default: "triangles").

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics including:
        - recall: Structure recall
        - density: Edge density
        - num_structures: Number of structures found

    Examples
    --------
    >>> metrics = summarize_batch_metrics(batch, golden, "triangles")
    >>> print(f"Recall: {metrics['recall']:.1%}")
    >>> print(f"Density: {metrics['density']:.2f}")
    """
    batch_structures = extract_structures_from_data(batch_data, structure_type)
    recall = compute_structure_recall(batch_structures, golden_structures)
    density = compute_edge_density(batch_data.edge_index, batch_data.num_nodes)

    return {
        "recall": recall,
        "density": density,
        "num_structures": len(batch_structures),
        "num_nodes": batch_data.num_nodes,
        "num_edges": batch_data.edge_index.shape[1] // 2,
    }
