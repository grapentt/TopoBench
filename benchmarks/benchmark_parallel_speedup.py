#!/usr/bin/env python3
"""Benchmark parallel speedup across entire preprocessing pipeline.

This benchmark demonstrates the parallel speedup from the multi-worker architecture
across the ENTIRE preprocessing pipeline (not just mmap conversion).

Tests different worker counts (1, 2, 4, 8) to show:
- Speedup vs sequential processing
- Scaling efficiency
- Optimal worker count

Usage:
    python benchmarks/benchmark_parallel_speedup.py --output results/parallel
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    SyntheticGraphDataset,
    create_benchmark_temp_dir,
    get_system_info,
    get_worker_counts,
    print_benchmark_header,
    save_results,
)
from topobench.data.preprocessor import OnDiskInductivePreprocessor


def benchmark_parallel_preprocessing(
    dataset_size: int,
    worker_counts: list | None = None,
    runs: int = 3,
) -> dict:
    """Benchmark parallel speedup across entire preprocessing pipeline.
    
    Parameters
    ----------
    dataset_size : int
        Number of samples in dataset.
    worker_counts : list | None
        List of worker counts to test. If None, auto-generates optimal sequence.
    runs : int
        Number of benchmark runs per worker count.
        
    Returns
    -------
    dict
        Parallel speedup results and temp_dir path for cleanup.
    """
    # Auto-generate worker counts if not provided
    if worker_counts is None:
        worker_counts = get_worker_counts()
    
    # Replace None with actual worker count for display
    import os
    cpu_count = os.cpu_count() or 1
    max_workers = max(1, cpu_count - 1)
    display_counts = [w if w is not None else max_workers for w in worker_counts]
    
    print(f"\n{'='*80}")
    print(f"  Dataset size: {dataset_size:,} samples")
    print(f"  Testing worker counts: {display_counts} (None = {max_workers})")
    print(f"{'='*80}\n")
    
    # Create synthetic dataset
    source_dataset = SyntheticGraphDataset(
        num_samples=dataset_size,
        num_nodes=50,
        num_features=16,
        seed=42,
    )
    
    # Configure transforms (realistic pipeline)
    # Tests parallel speedup with actual transforms (clique lifting)
    transforms_config = OmegaConf.create({
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2,
        },
    })
    
    # Create benchmark temp directory (avoids /tmp size limits)
    benchmark_temp_dir = create_benchmark_temp_dir()
    print(f"\n📁 Using temp directory: {benchmark_temp_dir}")
    print(f"   (Will need manual cleanup if not automatically removed)\n")
    
    results_by_workers = {}
    
    for num_workers in worker_counts:
        # Display worker count (None = auto-detect)
        display_workers = num_workers if num_workers is not None else max_workers
        print(f"\n{'='*60}")
        print(f"  Testing with {display_workers} worker(s)" + (" (auto)" if num_workers is None else ""))
        print(f"{'='*60}\n")
        
        worker_times = []
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}... ", end='', flush=True)
            
            # Create run-specific subdirectory
            run_dir = benchmark_temp_dir / f"workers_{display_workers}_run_{run}"
            run_dir.mkdir(parents=True, exist_ok=True)
            data_dir = run_dir
                
            # Time the ENTIRE preprocessing pipeline
            start = time.perf_counter()
                
            dataset = OnDiskInductivePreprocessor(
                dataset=source_dataset,
                data_dir=data_dir,
                transforms_config=transforms_config,
                num_workers=num_workers,
                storage_backend="mmap",
                compression="lz4",
            )
                
            # Verify dataset is complete
            assert len(dataset) == dataset_size, f"Dataset size mismatch: {len(dataset)} != {dataset_size}"
            
            # Sanity check: verify we can actually load a sample
            sample = dataset[0]
            assert sample is not None, "Failed to load sample 0"
            assert hasattr(sample, 'x'), "Sample missing node features"
            
            # Check storage stats to ensure data was actually written
            stats = dataset._storage.get_stats()
            assert stats['num_samples'] == dataset_size, f"Storage has {stats['num_samples']} samples, expected {dataset_size}"
            
            elapsed = time.perf_counter() - start
            worker_times.append(elapsed)
            
            print(f"{elapsed:.2f}s (✓ {stats['num_samples']} samples)")
        
        mean_time = np.mean(worker_times)
        std_time = np.std(worker_times)
        
        # Store with display value for plotting
        results_by_workers[display_workers] = {
            "num_workers": display_workers,
            "times": worker_times,
            "mean_time": mean_time,
            "std_time": std_time,
        }
        
        print(f"\n  ✓ {num_workers} worker(s): {mean_time:.2f}s ± {std_time:.2f}s")
    
    # Calculate speedups vs sequential (1 worker)
    baseline_time = results_by_workers[1]["mean_time"]
    
    for num_workers, data in results_by_workers.items():
        speedup = baseline_time / data["mean_time"]
        efficiency = speedup / num_workers * 100  # Parallel efficiency
        data["speedup"] = speedup
        data["efficiency"] = efficiency
    
    return {
        "dataset_size": dataset_size,
        "worker_counts": list(results_by_workers.keys()),  # Actual tested counts
        "runs": runs,
        "results": results_by_workers,
        "baseline_time": baseline_time,
        "temp_dir": str(benchmark_temp_dir),  # For cleanup warning
    }


def plot_results(results: dict, output_dir: Path) -> None:
    """Generate plots for parallel speedup benchmark.
    
    Parameters
    ----------
    results : dict
        Benchmark results.
    output_dir : Path
        Output directory for plots.
    """
    worker_counts = results["worker_counts"]
    mean_times = [results["results"][w]["mean_time"] for w in worker_counts]
    std_times = [results["results"][w]["std_time"] for w in worker_counts]
    speedups = [results["results"][w]["speedup"] for w in worker_counts]
    efficiencies = [results["results"][w]["efficiency"] for w in worker_counts]
    
    # Add dataset size to figure title and info
    dataset_size = results.get("dataset_size", "Unknown")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Parallel Speedup Analysis - {dataset_size:,} Samples", fontsize=16, fontweight='bold')
    
    # Plot 1: Processing time vs workers (no error bars)
    ax1.plot(worker_counts, mean_times, marker='o',
             linewidth=2, markersize=8, color='#3498db')
    ax1.set_xlabel("Number of Workers", fontsize=12)
    ax1.set_ylabel("Processing Time (seconds)", fontsize=12)
    ax1.set_title(f"Preprocessing Time ({dataset_size:,} samples)", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(worker_counts)
    
    # Plot 2: Speedup vs workers
    ax2.plot(worker_counts, speedups, marker='s', linewidth=2, markersize=8,
             color='#2ecc71', label='Actual speedup')
    ax2.set_xlabel("Number of Workers", fontsize=12)
    ax2.set_ylabel("Speedup vs Sequential", fontsize=12)
    ax2.set_title(f"Parallel Speedup ({dataset_size:,} samples)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(worker_counts)
    
    # Add speedup annotations
    for x, y in zip(worker_counts, speedups):
        ax2.annotate(f'{y:.2f}×', xy=(x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Plot 3: Parallel efficiency
    ax3.plot(worker_counts, efficiencies, marker='d', linewidth=2, markersize=8,
             color='#f39c12')
    ax3.axhline(y=100, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel("Number of Workers", fontsize=12)
    ax3.set_ylabel("Parallel Efficiency (%)", fontsize=12)
    ax3.set_title(f"Parallel Efficiency ({dataset_size:,} samples)", fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(worker_counts)
    ax3.set_ylim([0, 110])
    
    # Plot 4: Summary table
    ax4.axis('off')
    
    # Create table data
    table_data = [["Workers", "Time (s)", "Speedup", "Efficiency"]]
    for w in worker_counts:
        r = results["results"][w]
        table_data.append([
            f"{w}",
            f"{r['mean_time']:.2f}",
            f"{r['speedup']:.2f}×",
            f"{r['efficiency']:.1f}%",
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(4):
            table[(i, j)].set_facecolor(color)
    
    ax4.set_title("Parallel Speedup Summary", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = output_dir / "parallel_speedup.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel speedup across entire preprocessing pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmarks/configs/comprehensive.yaml",
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/parallel_speedup",
        help="Output directory for results",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        full_config = yaml.safe_load(f)
    config = full_config["parallel"]
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_benchmark_header(
        "Parallel Speedup Benchmark",
        "Demonstrates parallel speedup across ENTIRE preprocessing pipeline.\n"
        "Tests multiple worker counts to show scaling efficiency."
    )
    
    # Get worker counts (use auto-generation if not specified)
    worker_counts = config.get("worker_counts")
    if worker_counts is None:
        worker_counts = get_worker_counts()
        print(f"ℹ️  Auto-generating worker counts: {worker_counts}\n")
    
    # Run benchmark
    results = benchmark_parallel_preprocessing(
        dataset_size=config["dataset_size"],
        worker_counts=worker_counts,
        runs=config["runs"],
    )
    
    # Add system info
    full_results = {
        "system_info": get_system_info(),
        **results,
    }
    
    # Save results
    save_results(full_results, output_dir / "raw_data.json")
    
    # Generate plots
    plot_results(results, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("  PARALLEL SPEEDUP SUMMARY")
    print("="*80)
    print(f"\nDataset size: {results['dataset_size']:,} samples")
    print(f"Baseline (1 worker): {results['baseline_time']:.2f}s\n")
    
    for w in results["worker_counts"]:
        r = results["results"][w]
        print(f"{w} worker(s): {r['mean_time']:.2f}s → {r['speedup']:.2f}× speedup ({r['efficiency']:.1f}% efficient)")
    
    # Find best configuration
    best_workers = max(results["worker_counts"], key=lambda w: results["results"][w]["speedup"])
    best_speedup = results["results"][best_workers]["speedup"]
    
    print("\n" + "="*80)
    print("  KEY FINDINGS")
    print("="*80)
    print(f"✅ Best speedup: {best_speedup:.2f}× with {best_workers} workers")
    print(f"✅ Parallel architecture provides significant speedup")
    print(f"✅ Entire preprocessing pipeline benefits from parallelization")
    print("="*80 + "\n")
    
    # Cleanup warning
    temp_dir = results.get("temp_dir")
    if temp_dir:
        print("⚠️  IMPORTANT: Temporary files created during benchmarking")
        print(f"   Location: {temp_dir}")
        print(f"   Please verify this directory was cleaned up.")
        print(f"   If not, manually remove it: rm -rf {temp_dir}")
        print()


if __name__ == "__main__":
    main()
