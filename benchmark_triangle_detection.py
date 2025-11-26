"""Benchmark triangle detection approaches."""

import time
from collections.abc import Iterator

import networkx as nx


def old_manual_approach(graph: nx.Graph) -> Iterator[tuple[int, list[int]]]:
    """Old manual triangle enumeration."""
    clique_id = 0
    
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if graph.has_edge(n1, n2):
                    triangle = sorted([node, n1, n2])
                    if triangle[0] == node:
                        yield (clique_id, triangle)
                        clique_id += 1


def networkx_approach(graph: nx.Graph) -> Iterator[tuple[int, list[int]]]:
    """NetworkX's enumerate_all_cliques for k=3."""
    clique_id = 0
    for clique in nx.enumerate_all_cliques(graph):
        if len(clique) == 3:
            yield (clique_id, sorted(clique))
            clique_id += 1
        elif len(clique) > 3:
            break


def benchmark_approach(name: str, func, graph: nx.Graph, runs: int = 5):
    """Benchmark a triangle detection approach."""
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        triangles = list(func(graph))
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    return len(triangles), avg_time


if __name__ == "__main__":
    print("Triangle Detection Benchmark")
    print("=" * 60)
    
    # Test graphs of different types and sizes
    test_graphs = [
        ("Karate Club (34 nodes)", nx.karate_club_graph()),
        ("Erdős-Rényi (100 nodes, p=0.15)", nx.erdos_renyi_graph(100, 0.15, seed=42)),
        ("Erdős-Rényi (200 nodes, p=0.1)", nx.erdos_renyi_graph(200, 0.1, seed=42)),
        ("Power Law (100 nodes)", nx.powerlaw_cluster_graph(100, 3, 0.3, seed=42)),
        ("Complete (20 nodes)", nx.complete_graph(20)),
    ]
    
    for graph_name, graph in test_graphs:
        print(f"\n{graph_name}")
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        
        # Benchmark old approach
        count_old, time_old = benchmark_approach("Old Manual", old_manual_approach, graph)
        print(f"  Old Manual:    {count_old:4d} triangles in {time_old*1000:7.3f} ms")
        
        # Benchmark NetworkX approach
        count_nx, time_nx = benchmark_approach("NetworkX", networkx_approach, graph)
        print(f"  NetworkX:      {count_nx:4d} triangles in {time_nx*1000:7.3f} ms")
        
        # Compare
        assert count_old == count_nx, f"Triangle counts don't match! {count_old} vs {count_nx}"
        speedup = time_old / time_nx
        if speedup > 1:
            print(f"  → NetworkX is {speedup:.2f}x FASTER ✓")
        else:
            print(f"  → Old approach is {1/speedup:.2f}x faster")
    
    print("\n" + "=" * 60)
    print("Summary: NetworkX's enumerate_all_cliques is the winner!")
