"""Plotting utilities for benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_memory_plots(results: dict, output_dir: Path) -> None:
    """Generate memory profiling plots.
    
    Parameters
    ----------
    results : dict
        Memory profiling results
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_sizes = results["dataset_sizes"]
    
    # Extract data
    inmemory_peaks = [results["inmemory"][s].get("peak_mb") for s in dataset_sizes 
                      if results["inmemory"][s].get("peak_mb")]
    ondisk_peaks = [results["ondisk"][s].get("peak_mb") for s in dataset_sizes 
                    if results["ondisk"][s].get("peak_mb")]
    
    valid_sizes_inmem = [s for s in dataset_sizes if results["inmemory"][s].get("peak_mb")]
    valid_sizes_ondisk = [s for s in dataset_sizes if results["ondisk"][s].get("peak_mb")]
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle("Memory Profiling: O(1) vs O(N)", fontsize=16, fontweight='bold')
    
    # Plot 1: Absolute memory usage
    if inmemory_peaks:
        ax1.plot(valid_sizes_inmem, inmemory_peaks, 'o-', label='InMemory', 
                linewidth=2, markersize=8, color='#e74c3c')
    if ondisk_peaks:
        ax1.plot(valid_sizes_ondisk, ondisk_peaks, 's-', label='OnDisk', 
                linewidth=2, markersize=8, color='#27ae60')
    
    ax1.set_xlabel('Dataset Size (samples)', fontsize=12)
    ax1.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax1.set_title('Absolute Memory Usage', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory per sample
    if inmemory_peaks and len(valid_sizes_inmem) > 0:
        inmem_per_sample = [results["inmemory"][s].get("per_sample_kb") 
                           for s in valid_sizes_inmem if results["inmemory"][s].get("per_sample_kb")]
        if inmem_per_sample:
            ax2.plot(valid_sizes_inmem[:len(inmem_per_sample)], inmem_per_sample, 
                    'o-', label='InMemory', linewidth=2, markersize=8, color='#e74c3c')
    
    if ondisk_peaks and len(valid_sizes_ondisk) > 0:
        ondisk_per_sample = [results["ondisk"][s].get("per_sample_kb") 
                            for s in valid_sizes_ondisk if results["ondisk"][s].get("per_sample_kb")]
        if ondisk_per_sample:
            ax2.plot(valid_sizes_ondisk[:len(ondisk_per_sample)], ondisk_per_sample, 
                    's-', label='OnDisk', linewidth=2, markersize=8, color='#27ae60')
    
    ax2.set_xlabel('Dataset Size (samples)', fontsize=12)
    ax2.set_ylabel('Memory per Sample (KB)', fontsize=12)
    ax2.set_title('Memory per Sample', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory savings ratio
    valid_both = [s for s in dataset_sizes 
                  if results["inmemory"][s].get("peak_mb") and results["ondisk"][s].get("peak_mb")]
    
    if valid_both:
        savings = [results["inmemory"][s]["peak_mb"] / results["ondisk"][s]["peak_mb"] 
                  for s in valid_both]
        ax3.bar(range(len(valid_both)), savings, color='#3498db', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(valid_both)))
        ax3.set_xticklabels([f"{s:,}" for s in valid_both], rotation=45)
        ax3.set_xlabel('Dataset Size (samples)', fontsize=12)
        ax3.set_ylabel('Memory Savings (×)', fontsize=12)
        ax3.set_title('Memory Savings Ratio (InMemory / OnDisk)', fontsize=14, fontweight='bold')
        ax3.axhline(y=1.0, color='red', linestyle='--', label='No savings', alpha=0.5)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Log-log scaling
    if len(valid_sizes_inmem) >= 2 and len(valid_sizes_ondisk) >= 2:
        ax4.loglog(valid_sizes_inmem, inmemory_peaks, 'o-', label='InMemory (O(N))', 
                  linewidth=2, markersize=8, color='#e74c3c')
        ax4.loglog(valid_sizes_ondisk, ondisk_peaks, 's-', label='OnDisk (O(1))', 
                  linewidth=2, markersize=8, color='#27ae60')
        
        # Add trend lines
        if "scaling_analysis" in results:
            sa = results["scaling_analysis"]
            sizes_array = np.array(valid_sizes_inmem)
            inmem_trend = sa["inmemory_slope"] * sizes_array + sa["inmemory_intercept"]
            ondisk_trend = sa["ondisk_slope"] * sizes_array + sa["ondisk_intercept"]
            ax4.loglog(sizes_array, inmem_trend, '--', color='#c0392b', alpha=0.5, label='InMemory trend')
            ax4.loglog(sizes_array, ondisk_trend, '--', color='#229954', alpha=0.5, label='OnDisk trend')
        
        ax4.set_xlabel('Dataset Size (samples)', fontsize=12)
        ax4.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax4.set_title('Scaling Behavior (Log-Log)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plot_path = output_dir / "memory_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Plot saved: {plot_path}")


def generate_parallel_plots(results: dict, output_dir: Path) -> None:
    """Generate parallel speedup plots.
    
    Parameters
    ----------
    results : dict
        Parallel speedup results
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "speedup_analysis" not in results:
        print("  ⚠️  No speedup analysis to plot")
        return
    
    sa = results["speedup_analysis"]
    worker_counts = sorted(sa["speedups"].keys())
    speedups = [sa["speedups"][w] for w in worker_counts]
    efficiencies = [sa["efficiencies"][w] for w in worker_counts]
    
    # Get timing data
    times = [results["measurements"][w]["time_mean_sec"] for w in worker_counts 
             if w in results["measurements"] and "time_mean_sec" in results["measurements"][w]]
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle("Parallel Processing Speedup Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Speedup curve
    ax1.plot(worker_counts, speedups, 'o-', linewidth=2, markersize=10, 
            color='#3498db', label='Actual Speedup')
    ax1.set_xlabel('Number of Workers', fontsize=12)
    ax1.set_ylabel('Speedup (×)', fontsize=12)
    ax1.set_title('Parallel Speedup', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(worker_counts)
    
    # Plot 2: Parallel efficiency
    ax2.plot(worker_counts, [e * 100 for e in efficiencies], 'o-', 
            linewidth=2, markersize=10, color='#e74c3c')
    ax2.axhline(y=100, color='#95a5a6', linestyle='--', label='Ideal (100%)', alpha=0.7)
    ax2.axhline(y=50, color='#f39c12', linestyle=':', label='50% threshold', alpha=0.5)
    ax2.set_xlabel('Number of Workers', fontsize=12)
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax2.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(worker_counts)
    ax2.set_ylim([0, 110])
    
    # Plot 3: Execution time
    if times:
        ax3.bar(range(len(worker_counts)), times, color='#27ae60', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(worker_counts)))
        ax3.set_xticklabels(worker_counts)
        ax3.set_xlabel('Number of Workers', fontsize=12)
        ax3.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax3.set_title('Execution Time by Worker Count', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, (time_val, count) in enumerate(zip(times, worker_counts)):
            ax3.text(i, time_val, f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Throughput
    throughputs = [results["measurements"][w]["throughput_samples_per_sec"] 
                  for w in worker_counts if w in results["measurements"]]
    
    if throughputs:
        ax4.plot(worker_counts, throughputs, 'o-', linewidth=2, markersize=10, color='#9b59b6')
        ax4.set_xlabel('Number of Workers', fontsize=12)
        ax4.set_ylabel('Throughput (samples/sec)', fontsize=12)
        ax4.set_title('Processing Throughput', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(worker_counts)
        
        # Add values
        for w, t in zip(worker_counts, throughputs):
            ax4.text(w, t, f'{t:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / "speedup_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Plot saved: {plot_path}")
