"""Utilities for validation scripts demonstrating OOM vs on-disk success.

This module provides helpers for creating reproducible validation scripts
that demonstrate memory limitations of in-memory approaches and success
of on-disk approaches.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Callable

import psutil
import torch


class MemoryTracker:
    """Track memory usage during execution.

    Tracks both RAM and GPU memory if CUDA is available.

    Examples
    --------
    >>> tracker = MemoryTracker()
    >>> with tracker:
    ...     # Your code here
    ...     data = torch.randn(1000, 1000)
    >>> print(f"Peak RAM: {tracker.peak_ram_mb:.0f}MB")
    >>> print(f"Peak GPU: {tracker.peak_gpu_mb:.0f}MB")
    """

    def __init__(self):
        """Initialize memory tracker."""
        self.process = psutil.Process(os.getpid())
        self.peak_ram_mb = 0.0
        self.peak_gpu_mb = 0.0
        self.start_ram_mb = 0.0
        self.start_gpu_mb = 0.0
        self.measurements = []

    def __enter__(self):
        """Start tracking."""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.start_ram_mb = self.process.memory_info().rss / 1024 / 1024
        self.start_gpu_mb = self._get_gpu_memory()
        self.peak_ram_mb = self.start_ram_mb
        self.peak_gpu_mb = self.start_gpu_mb
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking."""
        self.update()
        return False

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1024 / 1024

    def update(self):
        """Update peak memory measurements."""
        current_ram = self.process.memory_info().rss / 1024 / 1024
        current_gpu = self._get_gpu_memory()
        
        self.peak_ram_mb = max(self.peak_ram_mb, current_ram)
        self.peak_gpu_mb = max(self.peak_gpu_mb, current_gpu)
        
        self.measurements.append({
            "ram_mb": current_ram,
            "gpu_mb": current_gpu,
            "timestamp": time.time()
        })

    def get_delta_ram(self) -> float:
        """Get RAM increase since start in MB."""
        return self.peak_ram_mb - self.start_ram_mb

    def get_delta_gpu(self) -> float:
        """Get GPU memory increase since start in MB."""
        return self.peak_gpu_mb - self.start_gpu_mb

    def report(self, prefix: str = ""):
        """Print memory usage report."""
        print(f"\n{prefix}Memory Usage Report:")
        print(f"  RAM: {self.start_ram_mb:.0f}MB → {self.peak_ram_mb:.0f}MB "
              f"(Δ {self.get_delta_ram():.0f}MB)")
        if torch.cuda.is_available():
            print(f"  GPU: {self.start_gpu_mb:.0f}MB → {self.peak_gpu_mb:.0f}MB "
                  f"(Δ {self.get_delta_gpu():.0f}MB)")


def calculate_oom_params(
    max_ram_gb: float = 4.0,
    approach: str = "inductive",
    structure_type: str = "triangles"
) -> dict:
    """Calculate dataset parameters guaranteed to cause OOM.

    This function calculates dataset sizes that will definitely exceed
    the specified RAM limit when using in-memory preprocessing.

    Parameters
    ----------
    max_ram_gb : float
        Maximum RAM available for testing (default: 4GB).
    approach : str
        Either "inductive" or "transductive".
    structure_type : str
        Type of structures: "triangles" (3-cliques) or "4-cliques".

    Returns
    -------
    dict
        Parameters for dataset creation that will cause OOM.

    Examples
    --------
    >>> params = calculate_oom_params(max_ram_gb=4.0, approach="inductive")
    >>> print(f"Use {params['num_graphs']} graphs with {params['nodes_per_graph']} nodes")

    Notes
    -----
    Formula for memory estimation:
    - Inductive: num_graphs × nodes × avg_degree^k × 12 bytes
    - Transductive: nodes × avg_degree^k × 12 bytes
    Where k=2 for triangles, k=3 for 4-cliques
    """
    # Conservative estimate: use 70% of available RAM for structures
    target_gb = max_ram_gb * 0.7
    bytes_per_structure = 12  # structure_id + node_ids
    
    # Determine k based on structure type
    k = 2 if structure_type == "triangles" else 3
    
    if approach == "inductive":
        # Multiple graphs scenario
        # Target: make num_graphs × nodes × degree^k × 12 > target_bytes
        
        # Use reasonable graph sizes
        nodes_per_graph = 100
        avg_degree = 20  # High degree to generate many structures
        
        # Structures per graph ≈ nodes × degree^k
        structures_per_graph = nodes_per_graph * (avg_degree ** k)
        bytes_per_graph = structures_per_graph * bytes_per_structure
        
        # How many graphs to exceed target?
        num_graphs = int((target_gb * 1e9) / bytes_per_graph * 1.5)  # 1.5x to guarantee OOM
        
        return {
            "num_graphs": max(100, num_graphs),
            "nodes_per_graph": nodes_per_graph,
            "avg_degree": avg_degree,
            "k_value": avg_degree // 2,  # For Watts-Strogatz
            "p_rewire": 0.3,
            "estimated_structures": num_graphs * structures_per_graph,
            "estimated_memory_gb": (num_graphs * bytes_per_graph) / 1e9
        }
    
    else:  # transductive
        # Single large graph
        # Target: make nodes × degree^k × 12 > target_bytes
        
        avg_degree = 50  # High degree for many structures
        
        # Structures ≈ nodes × degree^k
        structures_per_node = avg_degree ** k
        bytes_per_node = structures_per_node * bytes_per_structure
        
        # How many nodes to exceed target?
        num_nodes = int((target_gb * 1e9) / bytes_per_node * 1.5)  # 1.5x to guarantee OOM
        
        return {
            "num_nodes": max(10000, num_nodes),
            "avg_degree": avg_degree,
            "k_value": avg_degree // 2,
            "p_rewire": 0.3,
            "estimated_structures": num_nodes * structures_per_node,
            "estimated_memory_gb": (num_nodes * bytes_per_node) / 1e9
        }


def expect_oom(func: Callable, timeout_seconds: int = 300) -> tuple[bool, str]:
    """Execute function expecting it to run out of memory.

    Parameters
    ----------
    func : Callable
        Function to execute (should cause OOM).
    timeout_seconds : int
        Maximum time to wait before considering it hung (default: 300s).

    Returns
    -------
    tuple[bool, str]
        (oom_occurred, message) where oom_occurred is True if OOM happened.

    Examples
    --------
    >>> def will_oom():
    ...     x = torch.randn(100000, 100000)  # Too large
    >>> oom_occurred, msg = expect_oom(will_oom)
    >>> assert oom_occurred, "Expected OOM but didn't happen!"
    """
    try:
        func()
        return False, "Function completed without OOM (unexpected)"
    except (MemoryError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cannot allocate" in error_msg:
            return True, f"OOM occurred as expected: {type(e).__name__}"
        else:
            return False, f"Different error occurred: {e}"
    except Exception as e:
        return False, f"Unexpected exception: {type(e).__name__}: {e}"


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str, width: int = 80):
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def print_result(success: bool, message: str, indent: int = 2):
    """Print a formatted result message."""
    symbol = "✓" if success else "✗"
    indent_str = " " * indent
    print(f"{indent_str}{symbol} {message}")
