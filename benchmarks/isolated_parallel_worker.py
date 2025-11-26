#!/usr/bin/env python3
"""Parallel benchmark worker with deep debugging - runs in isolated subprocess."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import psutil
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import SyntheticGraphDataset
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductivePreprocessor


def measure_parallel_with_debug(num_samples: int, num_workers: int, batch_size: int, n_runs: int) -> dict:
    """Measure parallel speedup with comprehensive debugging.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to preprocess
    num_workers : int
        Number of parallel workers
    batch_size : int
        Batch size
    n_runs : int
        Number of runs
        
    Returns
    -------
    dict
        Timing results with debug info
    """
    import os
    import contextlib
    
    times = []
    debug_info = {
        "overhead_breakdown": [],
        "worker_stats": [],
        "process_info": [],
        "performance_analysis": {},
    }
    
    # Track CPU info
    cpu_count = os.cpu_count()
    debug_info["system"] = {
        "cpu_count": cpu_count,
        "cpu_count_logical": cpu_count,
        "num_workers_requested": num_workers,
        "batch_size": batch_size,
    }
    
    for run_idx in range(n_runs):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Track detailed timing
            timing = {
                "total": 0,
                "dataset_creation": 0,
                "preprocessing": 0,
                "process_spawn_overhead": 0,
                "work_time": 0,
            }
            
            # Track process metrics
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)
            
            # Create source dataset
            t0 = time.perf_counter()
            source_dataset = SyntheticGraphDataset(
                num_samples=num_samples,
                num_nodes=50,
                num_features=16,
            )
            timing["dataset_creation"] = time.perf_counter() - t0
            
            # Preprocess with timing hooks
            t1 = time.perf_counter()
            
            # Redirect stdout to stderr to avoid JSON contamination
            old_stdout = sys.stdout
            try:
                sys.stdout = sys.stderr
                
                # Intercept to track preprocessing details
                # NOTE: storage_backend="mmap" now uses parallel conversion!
                # OPTIMIZATION: Use larger batch_size to reduce pickling/IPC overhead
                effective_batch_size = max(batch_size, 500)  # Minimum 500 for efficiency
                dataset = OnDiskInductivePreprocessor(
                    dataset=source_dataset,
                    data_dir=tmpdir,
                    transforms_config=None,
                    num_workers=num_workers,
                    batch_size=effective_batch_size,
                    cache_size=0,
                    storage_backend="mmap",  # Enable parallel mmap conversion!
                    compression="lz4",  # Fast compression
                )
            finally:
                sys.stdout = old_stdout
            
            t2 = time.perf_counter()
            timing["preprocessing"] = t2 - t1
            timing["total"] = t2 - t0
            timing["work_time"] = timing["preprocessing"]
            
            # Estimate overhead
            if num_workers > 1:
                # Rough estimate: spawn + IPC overhead
                timing["process_spawn_overhead"] = max(0, timing["total"] - timing["work_time"])
            
            mem_after = process.memory_info().rss / (1024 * 1024)
            
            times.append(timing["total"])
            
            # Collect debug info for this run
            if run_idx == 0:  # Detailed stats for first run only
                samples_per_worker = num_samples / num_workers if num_workers > 0 else num_samples
                time_per_sample_ms = (timing["preprocessing"] / num_samples) * 1000
                
                debug_info["overhead_breakdown"].append({
                    "run": run_idx,
                    "dataset_creation_sec": timing["dataset_creation"],
                    "preprocessing_sec": timing["preprocessing"],
                    "total_sec": timing["total"],
                    "overhead_pct": (timing["dataset_creation"] / timing["total"]) * 100 if timing["total"] > 0 else 0,
                    "samples_per_worker": samples_per_worker,
                    "time_per_sample_ms": time_per_sample_ms,
                    "memory_delta_mb": mem_after - mem_before,
                })
                
                debug_info["process_info"].append({
                    "num_workers": num_workers,
                    "samples": num_samples,
                    "batch_size": batch_size,
                    "samples_per_worker": samples_per_worker,
                    "batches_per_worker": samples_per_worker / batch_size if batch_size > 0 else 0,
                })
    
    times_array = torch.tensor(times)
    
    # Calculate worker efficiency
    if num_workers > 1:
        # Estimate ideal time (assuming perfect scaling)
        sequential_time = times[0] if len(times) > 0 else 0
        ideal_parallel_time = sequential_time / num_workers
        actual_time = times_array.mean().item()
        parallel_efficiency = (ideal_parallel_time / actual_time) * 100 if actual_time > 0 else 0
    else:
        parallel_efficiency = 100.0
    
    # Add worker stats
    debug_info["worker_stats"].append({
        "num_workers": num_workers,
        "samples_per_worker": num_samples / num_workers if num_workers > 0 else num_samples,
        "parallel_efficiency_pct": parallel_efficiency,
    })
    
    # Performance analysis
    avg_time = times_array.mean().item()
    time_per_sample = avg_time / num_samples if num_samples > 0 else 0
    
    debug_info["performance_analysis"] = {
        "avg_time_sec": avg_time,
        "time_per_sample_ms": time_per_sample * 1000,
        "throughput_samples_per_sec": num_samples / avg_time if avg_time > 0 else 0,
        "parallel_efficiency_pct": parallel_efficiency,
        "estimated_overhead_ms": 50 if num_workers > 1 else 0,  # Rough estimate
        "work_dominated": time_per_sample > 0.010,  # >10ms per sample
        "overhead_dominated": time_per_sample < 0.002,  # <2ms per sample
    }
    
    # Bottleneck diagnosis
    if time_per_sample < 0.002:  # < 2ms per sample
        bottleneck = "overhead"
        suggestion = f"Dataset too small. Increase to {num_samples * 10:,} samples or use larger graphs."
    elif parallel_efficiency < 50 and num_workers > 1:
        bottleneck = "poor_scaling"
        suggestion = f"Poor parallel scaling. Check for serialization bottlenecks or increase batch size."
    elif time_per_sample < 0.005:  # < 5ms per sample
        bottleneck = "borderline"
        suggestion = f"Work time close to overhead. Use --scale 5-10 for better results."
    else:
        bottleneck = "none"
        suggestion = "Performance looks good for current dataset size."
    
    debug_info["performance_analysis"]["bottleneck"] = bottleneck
    debug_info["performance_analysis"]["suggestion"] = suggestion
    
    return {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "time_mean_sec": times_array.mean().item(),
        "time_std_sec": times_array.std().item(),
        "time_min_sec": times_array.min().item(),
        "time_max_sec": times_array.max().item(),
        "throughput_samples_per_sec": num_samples / times_array.mean().item(),
        "raw_times": times,
        "debug_info": debug_info,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-runs', type=int, default=3)
    
    args = parser.parse_args()
    
    try:
        result = measure_parallel_with_debug(
            args.num_samples,
            args.num_workers,
            args.batch_size,
            args.n_runs,
        )
        
        # Output JSON to stdout (must be ONLY thing on stdout)
        print(json.dumps(result), flush=True)
        
    except Exception as e:
        import traceback
        # Write error to stderr, not stdout
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
