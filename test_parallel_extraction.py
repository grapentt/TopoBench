#!/usr/bin/env python
"""Quick test to demonstrate parallel extraction."""

import time
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.datasets import adapt_dataset


class SimpleDataset(InMemoryDataset):
    """Simple dataset for testing."""
    
    def __init__(self, root, num_samples=100):
        self.num_samples = num_samples
        super().__init__(root)
        self.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data_list = []
        for i in range(self.num_samples):
            x = torch.randn(10, 8)
            edge_index = torch.randint(0, 10, (2, 20))
            y = torch.tensor([i % 5])
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        
        torch.save(self.collate(data_list), self.processed_paths[0])


def main():
    print("=" * 70)
    print("Parallel Extraction Demonstration")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test dataset
        print("\n📦 Creating test dataset (500 samples)...")
        dataset = SimpleDataset(str(tmpdir / "source"), num_samples=500)
        print(f"✓ Dataset created: {len(dataset)} samples")
        
        # Test 1: Sequential extraction
        print("\n⏱️  Test 1: Sequential Extraction (1 worker)")
        print("-" * 70)
        start = time.time()
        adapted_seq = adapt_dataset(
            dataset,
            root=str(tmpdir / "adapted_seq"),
            extraction_workers=1,  # Sequential
            verbose=True
        )
        time_seq = time.time() - start
        print(f"⏱️  Sequential extraction time: {time_seq:.3f}s")
        
        # Test 2: Parallel extraction
        print("\n⚡ Test 2: Parallel Extraction (all cores - 1)")
        print("-" * 70)
        start = time.time()
        adapted_par = adapt_dataset(
            dataset,
            root=str(tmpdir / "adapted_par"),
            extraction_workers=None,  # Parallel (all cores - 1)
            verbose=True
        )
        time_par = time.time() - start
        print(f"⏱️  Parallel extraction time: {time_par:.3f}s")
        
        # Results
        speedup = time_seq / time_par
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Sequential:  {time_seq:.3f}s")
        print(f"Parallel:    {time_par:.3f}s")
        print(f"Speedup:     {speedup:.2f}×")
        print("=" * 70)
        
        if speedup >= 2.0:
            print(f"✅ Excellent speedup! ({speedup:.2f}×)")
        elif speedup >= 1.5:
            print(f"✅ Good speedup! ({speedup:.2f}×)")
        elif speedup >= 1.0:
            print(f"✅ Modest speedup ({speedup:.2f}×)")
        else:
            print(f"⚠️  Parallel overhead dominates ({speedup:.2f}×)")
        
        print("\n✓ Both adapted datasets ready for use!")
        print(f"  Sequential: {len(adapted_seq)} samples")
        print(f"  Parallel:   {len(adapted_par)} samples")


if __name__ == "__main__":
    main()
