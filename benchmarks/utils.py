"""Common utilities for benchmarking suite.

Provides shared infrastructure for all benchmarks including:
- System information capture
- Statistical analysis utilities
- Progress tracking and logging
- Result serialization
- Dataset generation
"""

import gc
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch_geometric.data import Data

# Register this module in sys.modules for pickling compatibility with multiprocessing
# This ensures worker processes can find SyntheticGraphDataset when unpickling
if __name__ != "__main__":
    sys.modules["benchmarks.utils"] = sys.modules[__name__]


def get_system_info() -> dict[str, Any]:
    """Capture system information for reproducibility.

    Returns
    -------
    dict
        System information including CPU, memory, Python, PyTorch versions.
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def get_worker_counts() -> list[int | None]:
    """Get optimal worker counts for parallel benchmarks.
    
    Returns sequence: 1, 2, 4, 8, 16, ..., and max_cores-1
    Example: For 8 cores → [1, 2, 4, None] where None=7
    
    Returns
    -------
    list[int | None]
        Worker counts to test, with None meaning (cores - 1).
    """
    cpu_count = os.cpu_count() or 1
    max_workers = max(1, cpu_count - 1)
    
    # Generate powers of 2: 1, 2, 4, 8, 16, ...
    worker_counts = []
    w = 1
    while w <= max_workers:
        worker_counts.append(w)
        w *= 2
    
    # If max_workers is not a power of 2, add it as None (auto-detect)
    if worker_counts[-1] != max_workers:
        worker_counts.append(None)
    
    return worker_counts


def get_single_worker_count() -> None:
    """Get worker count for single-config benchmarks.
    
    Returns None to use maximum available cores - 1.
    
    Returns
    -------
    None
        Indicates auto-detect optimal worker count.
    """
    return None


def create_benchmark_temp_dir() -> Path:
    """Create temporary directory for benchmarks in repo (not /tmp).
    
    Uses ./benchmark_tmp/ to avoid /tmp size limitations.
    User must manually clean up after benchmarks.
    
    Returns
    -------
    Path
        Path to temporary directory.
    """
    # Create in current directory to avoid /tmp size issues
    benchmark_tmp = Path.cwd() / "benchmark_tmp"
    benchmark_tmp.mkdir(parents=True, exist_ok=True)
    
    # Create unique subdirectory
    temp_subdir = tempfile.mkdtemp(dir=benchmark_tmp, prefix="bench_")
    return Path(temp_subdir)


def cleanup_benchmark_temp_dir(temp_dir: Path) -> None:
    """Clean up benchmark temporary directory.
    
    Parameters
    ----------
    temp_dir : Path
        Temporary directory to remove.
    """
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def measure_memory(func, *args, **kwargs) -> tuple[Any, float, float]:
    """Measure memory usage of a function.

    Parameters
    ----------
    func : callable
        Function to measure.
    *args
        Positional arguments to pass to func.
    **kwargs
        Keyword arguments to pass to func.

    Returns
    -------
    tuple[Any, float, float]
        (result, peak_memory_mb, delta_memory_mb)
    """
    gc.collect()
    process = psutil.Process()
    
    # Baseline memory
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    # Run function
    result = func(*args, **kwargs)
    
    # Peak memory
    mem_after = process.memory_info().rss / (1024 * 1024)
    delta = mem_after - mem_before
    
    return result, mem_after, delta


def measure_time(func, *args, n_runs: int = 3, warmup: int = 1, **kwargs) -> dict[str, float]:
    """Measure execution time with multiple runs for statistical robustness.

    Parameters
    ----------
    func : callable
        Function to measure.
    *args
        Positional arguments to pass to func.
    n_runs : int
        Number of measurement runs (default: 3).
    warmup : int
        Number of warmup runs (default: 1).
    **kwargs
        Keyword arguments to pass to func.

    Returns
    -------
    dict[str, float]
        Statistics: mean, std, min, max, median (all in seconds).
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
        "raw_times": times,
    }


def compute_confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval for data.

    Parameters
    ----------
    data : list[float]
        Data points.
    confidence : float
        Confidence level (default: 0.95).

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound)
    """
    from scipy import stats
    
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)
    
    return mean - interval, mean + interval


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save benchmark results to JSON file.

    Parameters
    ----------
    results : dict
        Results dictionary.
    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {output_path}")


def format_bytes(bytes_val: float) -> str:
    """Format bytes into human-readable string.

    Parameters
    ----------
    bytes_val : float
        Number of bytes.

    Returns
    -------
    str
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string.

    Parameters
    ----------
    seconds : float
        Number of seconds.

    Returns
    -------
    str
        Formatted string (e.g., "1.5 min").
    """
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds/60:.2f} min"
    else:
        return f"{seconds/3600:.2f} hr"


class SyntheticGraphDataset(torch.utils.data.Dataset):
    """Generate synthetic graph datasets for benchmarking.
    
    Properly inherits from torch.utils.data.Dataset for compatibility
    with TopoBench PreProcessor.
    
    Uses deterministic generation for reproducibility.
    
    Parameters
    ----------
    num_samples : int
        Number of samples.
    num_nodes : int
        Average number of nodes per graph.
    num_features : int
        Number of node features.
    seed : int
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        num_samples: int,
        num_nodes: int = 50,
        num_features: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seed = seed
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Data:
        """Generate sample deterministically based on index."""
        # Bounds check - CRITICAL for iteration to work correctly!
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.num_samples}")
        
        # Use idx as seed for deterministic generation
        torch.manual_seed(self.seed + idx)
        import numpy as np
        np.random.seed(self.seed + idx)
        
        # Variable graph size
        n_nodes = self.num_nodes + (idx % 10)
        n_edges_undirected = n_nodes * 2  # Number of undirected edges (will become 2x directed)
        
        # Generate node features
        x = torch.randn(n_nodes, self.num_features)
        
        # Create edge_index ensuring: no self-loops, no duplicates
        # Use numpy for better random generation without reseeding
        edges = []
        edge_set = set()
        attempts = 0
        max_attempts = n_edges_undirected * 10
        
        while len(edge_set) < n_edges_undirected and attempts < max_attempts:
            src = np.random.randint(0, n_nodes)
            dst = np.random.randint(0, n_nodes)
            
            # Skip self-loops
            if src == dst:
                attempts += 1
                continue
            
            # Store edges in canonical form (min, max) to avoid duplicates
            edge_tuple = (min(src, dst), max(src, dst))
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                # Add both directions for undirected graph
                edges.append([src, dst])
                edges.append([dst, src])
            
            attempts += 1
        
        # Convert to tensor
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Fallback: create a simple ring (both directions)
            ring_edges = []
            for i in range(n_nodes):
                ring_edges.append([i, (i + 1) % n_nodes])
                ring_edges.append([(i + 1) % n_nodes, i])
            edge_index = torch.tensor(ring_edges, dtype=torch.long).t().contiguous()
        
        y = torch.tensor([idx % 10])
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def __reduce__(self):
        """Support pickling for multiprocessing.
        
        Returns class with full module path to ensure worker processes
        can find it when unpickling.
        """
        # Use full module path for robust pickling across processes
        return (
            _reconstruct_synthetic_dataset,
            (self.num_samples, self.num_nodes, self.num_features, self.seed),
        )


def _reconstruct_synthetic_dataset(num_samples: int, num_nodes: int, num_features: int, seed: int):
    """Reconstruct SyntheticGraphDataset from pickle (helper for __reduce__).
    
    This function is defined at module level to ensure it's always findable
    by worker processes during unpickling.
    
    Parameters
    ----------
    num_samples : int
        Number of samples.
    num_nodes : int
        Average number of nodes per graph.
    num_features : int
        Number of node features.
    seed : int
        Random seed.
    
    Returns
    -------
    SyntheticGraphDataset
        Reconstructed dataset instance.
    """
    return SyntheticGraphDataset(num_samples, num_nodes, num_features, seed)


def print_benchmark_header(name: str, description: str) -> None:
    """Print formatted benchmark header.

    Parameters
    ----------
    name : str
        Benchmark name.
    description : str
        Benchmark description.
    """
    print("\n" + "=" * 80)
    print(f"  {name}")
    print("=" * 80)
    print(f"{description}\n")
    
    # Print system info
    info = get_system_info()
    print("System Information:")
    print(f"  CPU: {info['processor']} ({info['cpu_count']} cores)")
    print(f"  RAM: {info['total_memory_gb']:.1f} GB")
    print(f"  Python: {info['python_version']}")
    print(f"  PyTorch: {info['pytorch_version']}")
    print()


def print_benchmark_results(results: dict[str, Any]) -> None:
    """Print formatted benchmark results.

    Parameters
    ----------
    results : dict
        Results dictionary.
    """
    print("\n" + "-" * 80)
    print("  RESULTS")
    print("-" * 80)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("-" * 80 + "\n")
