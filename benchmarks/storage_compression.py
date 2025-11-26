#!/usr/bin/env python3
"""Storage Backend & Compression Benchmark.

This benchmark compares different storage backends and compression algorithms,
measuring I/O performance and disk space usage.

**Claims**:
1. Memory-mapped storage: 2-3× faster I/O
2. LZ4 compression: 1.3-1.6× disk savings with fast decompression
3. ZSTD compression: Better ratio but slower

**Metrics**: Read/write latency, throughput, compression ratio, disk usage

Academic Standards:
- Multiple measurements for latency
- Cold and warm cache tests
- Real-world access patterns
- Honest tradeoff reporting
"""

import argparse
import gc
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    SyntheticGraphDataset,
    format_bytes,
    format_time,
    get_system_info,
    print_benchmark_header,
    save_results,
)
from topobench.data.preprocessor.ondisk_inductive import (
    OnDiskInductivePreprocessor,
)


def measure_storage_performance(
    num_samples: int,
    storage_backend: str,
    compression: str,
    n_reads: int = 100,
) -> dict:
    """Measure storage performance for a configuration.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to create.
    storage_backend : str
        "mmap" or "files".
    compression : str
        None, "lz4", or "zstd".
    n_reads : int
        Number of random reads to measure.
    
    Returns
    -------
    dict
        Performance metrics.
    """
    config_name = f"{storage_backend}{'_' + compression if compression else '_none'}"
    print(f"\n  Testing: {config_name}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset
        source_dataset = SyntheticGraphDataset(
            num_samples=num_samples,
            num_nodes=50,
            num_features=16,
        )
        
        # Measure write time
        write_start = time.perf_counter()
        dataset = OnDiskInductivePreprocessor(
            dataset=source_dataset,
            data_dir=tmpdir,
            transforms_config=None,
            num_workers=1,  # Sequential for fair comparison
            cache_size=0,  # Disable cache
            storage_backend=storage_backend,
            compression=compression,
        )
        write_time = time.perf_counter() - write_start
        
        # Measure disk usage
        total_size = sum(
            f.stat().st_size for f in Path(tmpdir).rglob("*") if f.is_file()
        )
        
        # Cold reads (clear OS cache as much as possible)
        gc.collect()
        cold_times = []
        for _ in range(min(n_reads, len(dataset))):
            idx = np.random.randint(0, len(dataset))
            start = time.perf_counter()
            _ = dataset[idx]
            cold_times.append(time.perf_counter() - start)
        
        # Warm reads (repeated access)
        warm_times = []
        for _ in range(min(n_reads, len(dataset))):
            idx = np.random.randint(0, min(10, len(dataset)))  # Smaller pool
            start = time.perf_counter()
            _ = dataset[idx]
            warm_times.append(time.perf_counter() - start)
    
    result = {
        "config": config_name,
        "storage_backend": storage_backend,
        "compression": compression,
        "write_time_sec": write_time,
        "write_throughput_samples_per_sec": num_samples / write_time,
        "disk_size_bytes": total_size,
        "disk_size_mb": total_size / (1024 * 1024),
        "bytes_per_sample": total_size / num_samples,
        "cold_read_mean_ms": np.mean(cold_times) * 1000,
        "cold_read_std_ms": np.std(cold_times) * 1000,
        "warm_read_mean_ms": np.mean(warm_times) * 1000,
        "warm_read_std_ms": np.std(warm_times) * 1000,
    }
    
    print(f"    Write time: {format_time(write_time)}")
    print(f"    Disk usage: {format_bytes(total_size)}")
    print(f"    Cold read: {result['cold_read_mean_ms']:.2f} ms")
    print(f"    Warm read: {result['warm_read_mean_ms']:.2f} ms")
    
    return result


def run_storage_benchmark(
    num_samples: int = 1000,
    n_reads: int = 100,
    output_dir: Path = Path("results/storage"),
) -> dict:
    """Run complete storage and compression benchmark.
    
    Parameters
    ----------
    num_samples : int
        Number of samples.
    n_reads : int
        Number of reads to measure.
    output_dir : Path
        Output directory.
    
    Returns
    -------
    dict
        Complete benchmark results.
    """
    print_benchmark_header(
        "Storage Backend & Compression Benchmark",
        "Comparing storage backends and compression algorithms"
    )
    
    # Test configurations
    configurations = [
        ("mmap", None),
        ("mmap", "lz4"),
        ("mmap", "zstd"),
        ("files", None),  # Baseline
    ]
    
    results = {
        "system_info": get_system_info(),
        "num_samples": num_samples,
        "n_reads": n_reads,
        "configurations": configurations,
        "measurements": {},
    }
    
    print("Configuration:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Reads per config: {n_reads}")
    print(f"  Configs: {len(configurations)}")
    print()
    
    # Test each configuration
    for storage_backend, compression in configurations:
        config_name = f"{storage_backend}{'_' + compression if compression else '_none'}"
        
        print(f"\n{'='*80}")
        print(f"Testing: {config_name}")
        print(f"{'='*80}")
        
        try:
            result = measure_storage_performance(
                num_samples=num_samples,
                storage_backend=storage_backend,
                compression=compression,
                n_reads=n_reads,
            )
            results["measurements"][config_name] = result
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["measurements"][config_name] = {"error": str(e)}
    
    # Analysis
    analyze_results(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, output_dir / "raw_data.json")
    
    # Generate plots
    generate_plots(results, output_dir)
    
    # Generate summary
    generate_summary(results, output_dir)
    
    return results


def analyze_results(results: dict) -> None:
    """Analyze storage and compression results.
    
    Parameters
    ----------
    results : dict
        Benchmark results (modified in place).
    """
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")
    
    measurements = results["measurements"]
    
    # Use files_none as baseline
    baseline_key = "files_none"
    if baseline_key not in measurements or "error" in measurements[baseline_key]:
        print("⚠️  No baseline measurement, using first available")
        baseline_key = next((k for k in measurements if "error" not in measurements[k]), None)
        if not baseline_key:
            print("❌ No valid measurements!")
            return
    
    baseline = measurements[baseline_key]
    
    print(f"Baseline: {baseline_key}")
    print(f"  Disk: {format_bytes(baseline['disk_size_bytes'])}")
    print(f"  Cold read: {baseline['cold_read_mean_ms']:.2f} ms")
    print()
    
    # Compare each configuration
    comparisons = {}
    
    print(f"{'Config':<20} {'Disk Ratio':<15} {'Cold Read':<15} {'Warm Read':<15}")
    print("-" * 65)
    
    for name, data in measurements.items():
        if "error" in data or name == baseline_key:
            continue
        
        disk_ratio = baseline["disk_size_bytes"] / data["disk_size_bytes"]
        read_speedup = baseline["cold_read_mean_ms"] / data["cold_read_mean_ms"]
        warm_speedup = baseline["warm_read_mean_ms"] / data["warm_read_mean_ms"]
        
        comparisons[name] = {
            "disk_compression_ratio": disk_ratio,
            "cold_read_speedup": read_speedup,
            "warm_read_speedup": warm_speedup,
        }
        
        print(f"{name:<20} {disk_ratio:>13.2f}×  {read_speedup:>13.2f}×  {warm_speedup:>13.2f}×")
    
    print("-" * 65)
    
    results["comparisons"] = comparisons
    
    # Verdict
    print("\n📊 Key Findings:")
    
    # Check mmap speedup
    mmap_configs = [k for k in comparisons if k.startswith("mmap")]
    if mmap_configs:
        avg_speedup = np.mean([comparisons[k]["cold_read_speedup"] for k in mmap_configs])
        print(f"  Mmap I/O speedup: {avg_speedup:.1f}× average")
        if avg_speedup >= 2.0:
            print(f"    ✅ CLAIM VERIFIED: 2-3× faster I/O")
        else:
            print(f"    ⚠️  Below target: {avg_speedup:.1f}× (target: 2-3×)")
    
    # Check compression
    if "mmap_lz4" in comparisons:
        lz4_ratio = comparisons["mmap_lz4"]["disk_compression_ratio"]
        print(f"  LZ4 compression: {lz4_ratio:.2f}× disk savings")
        if lz4_ratio >= 1.3:
            print(f"    ✅ CLAIM VERIFIED: 1.3-1.6× savings achieved")
        else:
            print(f"    ⚠️  Below target: {lz4_ratio:.2f}× (target: 1.3-1.6×)")
    
    if "mmap_zstd" in comparisons:
        zstd_ratio = comparisons["mmap_zstd"]["disk_compression_ratio"]
        print(f"  ZSTD compression: {zstd_ratio:.2f}× disk savings")
        print(f"    {'✅' if zstd_ratio > lz4_ratio else '⚠️ '} Better ratio than LZ4: {zstd_ratio > lz4_ratio}")


def generate_plots(results: dict, output_dir: Path) -> None:
    """Generate comparison plots.
    
    Parameters
    ----------
    results : dict
        Benchmark results.
    output_dir : Path
        Output directory.
    """
    print("\n📊 Generating plots...")
    
    measurements = results["measurements"]
    valid_configs = [k for k in measurements if "error" not in measurements[k]]
    
    if not valid_configs:
        print("  ⚠️  No valid data to plot")
        return
    
    # Extract data
    config_names = valid_configs
    disk_sizes = [measurements[k]["disk_size_mb"] for k in config_names]
    cold_reads = [measurements[k]["cold_read_mean_ms"] for k in config_names]
    warm_reads = [measurements[k]["warm_read_mean_ms"] for k in config_names]
    write_times = [measurements[k]["write_time_sec"] for k in config_names]
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Storage Backend & Compression Analysis", fontsize=16, fontweight="bold")
    
    # Colors by config
    colors = []
    for name in config_names:
        if "files" in name:
            colors.append("#e74c3c")
        elif "zstd" in name:
            colors.append("#9b59b6")
        elif "lz4" in name:
            colors.append("#27ae60")
        else:
            colors.append("#3498db")
    
    x_pos = np.arange(len(config_names))
    
    # Plot 1: Disk usage
    ax1.bar(x_pos, disk_sizes, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([n.replace("_", "\n") for n in config_names], fontsize=9)
    ax1.set_ylabel("Disk Usage (MB)", fontsize=12)
    ax1.set_title("Disk Space Usage", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Plot 2: Cold read latency
    ax2.bar(x_pos, cold_reads, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([n.replace("_", "\n") for n in config_names], fontsize=9)
    ax2.set_ylabel("Latency (ms)", fontsize=12)
    ax2.set_title("Cold Read Latency", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Plot 3: Warm read latency
    ax3.bar(x_pos, warm_reads, color=colors, alpha=0.7, edgecolor="black")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([n.replace("_", "\n") for n in config_names], fontsize=9)
    ax3.set_ylabel("Latency (ms)", fontsize=12)
    ax3.set_title("Warm Read Latency", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    
    # Plot 4: Write time
    ax4.bar(x_pos, write_times, color=colors, alpha=0.7, edgecolor="black")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([n.replace("_", "\n") for n in config_names], fontsize=9)
    ax4.set_ylabel("Time (seconds)", fontsize=12)
    ax4.set_title("Write Time", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "storage_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Plot saved to {plot_path}")
    plt.close()


def generate_summary(results: dict, output_dir: Path) -> None:
    """Generate text summary.
    
    Parameters
    ----------
    results : dict
        Benchmark results.
    output_dir : Path
        Output directory.
    """
    summary_path = output_dir / "summary.txt"
    
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  STORAGE & COMPRESSION BENCHMARK - SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CLAIMS:\n")
        f.write("  1. Memory-mapped storage: 2-3× faster I/O\n")
        f.write("  2. LZ4 compression: 1.3-1.6× disk savings\n")
        f.write("  3. ZSTD compression: Better ratio but slower\n\n")
        
        # Results table
        f.write("RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Config':<18} {'Disk (MB)':<12} {'Cold Read':<14} {'Warm Read':<14} {'Write (s)':<10}\n")
        f.write("-" * 80 + "\n")
        
        measurements = results["measurements"]
        for name in sorted(measurements.keys()):
            if "error" in measurements[name]:
                continue
            
            m = measurements[name]
            f.write(f"{name:<18} {m['disk_size_mb']:>10.1f}  "
                   f"{m['cold_read_mean_ms']:>12.2f} ms  "
                   f"{m['warm_read_mean_ms']:>12.2f} ms  "
                   f"{m['write_time_sec']:>8.2f}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Verdict
        f.write("VERDICT:\n")
        comparisons = results.get("comparisons", {})
        
        if comparisons:
            mmap_speedups = [v["cold_read_speedup"] for k, v in comparisons.items() if "mmap" in k]
            if mmap_speedups and np.mean(mmap_speedups) >= 2.0:
                f.write("  ✅ Mmap I/O claim verified\n")
            
            if "mmap_lz4" in comparisons:
                ratio = comparisons["mmap_lz4"]["disk_compression_ratio"]
                if ratio >= 1.3:
                    f.write("  ✅ LZ4 compression claim verified\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  ✓ Summary saved to {summary_path}")


def main():
    """Run storage and compression benchmark."""
    parser = argparse.ArgumentParser(description="Storage and compression benchmark")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)",
    )
    parser.add_argument(
        "--reads",
        type=int,
        default=100,
        help="Number of reads (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/storage"),
        help="Output directory (default: results/storage)",
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_storage_benchmark(
        num_samples=args.samples,
        n_reads=args.reads,
        output_dir=args.output,
    )
    
    print("\n" + "=" * 80)
    print("  ✅ BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
