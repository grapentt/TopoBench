#!/usr/bin/env python3
"""Memory benchmark worker - runs in isolated subprocess.

This worker runs a single memory measurement in complete isolation
to eliminate baseline pollution.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import psutil
import torch
from torch_geometric.data import Data, InMemoryDataset

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import SyntheticGraphDataset
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductivePreprocessor


class InMemorySyntheticDataset(InMemoryDataset):
    """InMemory dataset for comparison."""
    
    def __init__(self, num_samples: int, num_nodes: int = 50, num_features: int = 16):
        self._num_samples = num_samples
        self._num_nodes = num_nodes
        self._num_features = num_features
        super().__init__()
        
        # Generate all data
        data_list = []
        for i in range(num_samples):
            torch.manual_seed(42 + i)
            n = num_nodes + (i % 10)
            data_list.append(
                Data(
                    x=torch.randn(n, num_features),
                    edge_index=torch.randint(0, n, (2, n * 3)),
                    y=torch.tensor([i % 10]),
                )
            )
        
        self.data, self.slices = self.collate(data_list)
    
    def len(self) -> int:
        return self._num_samples
    
    def get(self, idx: int) -> Data:
        return super().get(idx)


def measure_memory(approach: str, num_samples: int, num_accesses: int) -> dict:
    """Measure memory in isolated process.
    
    Parameters
    ----------
    approach : str
        "inmemory" or "ondisk"
    num_samples : int
        Number of samples
    num_accesses : int
        Number of samples to access
        
    Returns
    -------
    dict
        Memory measurements
    """
    process = psutil.Process()
    
    # Clean baseline
    import gc
    for _ in range(3):
        gc.collect()
    
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    
    if approach == "inmemory":
        # Create InMemory dataset
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        dataset = InMemorySyntheticDataset(
            num_samples=num_samples,
            num_nodes=50,
            num_features=16,
        )
        
        mem_after_creation = process.memory_info().rss / (1024 * 1024)
        
        # Access samples
        for i in range(min(num_accesses, len(dataset))):
            _ = dataset[i]
        
        mem_after_access = process.memory_info().rss / (1024 * 1024)
        peak_mb = max(mem_after_creation, mem_after_access)
        
    else:  # ondisk
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_before = process.memory_info().rss / (1024 * 1024)
            
            source_dataset = SyntheticGraphDataset(
                num_samples=num_samples,
                num_nodes=50,
                num_features=16,
            )
            
            # Redirect stdout to stderr to avoid JSON contamination
            old_stdout = sys.stdout
            try:
                sys.stdout = sys.stderr
                
                dataset = OnDiskInductivePreprocessor(
                    dataset=source_dataset,
                    data_dir=tmpdir,
                    transforms_config=None,
                    num_workers=1,
                    cache_size=0,
                )
            finally:
                sys.stdout = old_stdout
            
            mem_after_creation = process.memory_info().rss / (1024 * 1024)
            
            # Access samples
            for i in range(min(num_accesses, len(dataset))):
                _ = dataset[i]
            
            mem_after_access = process.memory_info().rss / (1024 * 1024)
            peak_mb = max(mem_after_creation, mem_after_access)
    
    delta_mb = peak_mb - baseline_mb
    
    return {
        "baseline_mb": baseline_mb,
        "after_creation_mb": mem_after_creation,
        "after_access_mb": mem_after_access,
        "peak_mb": peak_mb,
        "delta_mb": delta_mb,
        "per_sample_kb": delta_mb * 1024 / num_samples,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', required=True, choices=['inmemory', 'ondisk'])
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--num-accesses', type=int, required=True)
    
    args = parser.parse_args()
    
    result = measure_memory(args.approach, args.num_samples, args.num_accesses)
    
    # Output JSON to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
