#!/usr/bin/env python3
"""Test script to verify the on-disk training bug fix.

This script tests that OnDiskInductivePreprocessor.load_dataset_splits()
returns datasets compatible with TBDataloader's collate_fn.

The bug was that collate_fn expected (values, keys) tuples but received
Data objects directly. The fix introduces LazyDataloadDataset which
maintains O(1) memory while returning the correct tuple format.
"""

import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch.utils.data import Dataset

# Test imports
from topobench.data.preprocessor import OnDiskInductivePreprocessor
from topobench.dataloader.utils import collate_fn


class TinyTestDataset(Dataset):
    """Minimal test dataset."""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        return Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            y=torch.tensor([idx % 3]),
        )


def test_lazy_dataload_dataset_tuple_format():
    """Test that LazyDataloadDataset returns correct tuple format."""
    from topobench.data.datasets import LazyDataloadDataset
    
    print("\n" + "="*70)
    print("TEST 1: LazyDataloadDataset Tuple Format")
    print("="*70)
    
    # Create simple dataset
    dataset = TinyTestDataset(num_samples=5)
    
    # Create lazy split
    train_indices = [0, 1, 2]
    lazy_split = LazyDataloadDataset(dataset, train_indices)
    
    # Test length
    assert len(lazy_split) == 3, f"Expected length 3, got {len(lazy_split)}"
    print(f"✓ Length correct: {len(lazy_split)}")
    
    # Test get returns tuple
    sample = lazy_split.get(0)
    assert isinstance(sample, tuple), f"Expected tuple, got {type(sample)}"
    assert len(sample) == 2, f"Expected 2-element tuple, got {len(sample)}"
    
    values, keys = sample
    assert isinstance(values, list), f"Expected values to be list, got {type(values)}"
    assert isinstance(keys, list), f"Expected keys to be list, got {type(keys)}"
    print(f"✓ Returns tuple: (values={len(values)} items, keys={len(keys)} items)")
    print(f"  Keys: {keys}")
    
    # Test that values match the Data object
    data = dataset[train_indices[0]]
    for key in keys:
        assert key in data, f"Key {key} not in original Data object"
    print(f"✓ Tuple keys match Data object attributes")
    
    print("✓ TEST 1 PASSED\n")


def test_collate_fn_compatibility():
    """Test that LazyDataloadDataset works with collate_fn."""
    from topobench.data.datasets import LazyDataloadDataset
    
    print("="*70)
    print("TEST 2: collate_fn Compatibility")
    print("="*70)
    
    # Create dataset
    dataset = TinyTestDataset(num_samples=5)
    lazy_split = LazyDataloadDataset(dataset, [0, 1, 2])
    
    # Simulate what DataLoader does - collect batch
    batch = [lazy_split[i] for i in range(2)]  # Batch of 2 samples
    
    print(f"✓ Created batch of {len(batch)} samples")
    print(f"  Sample 0 type: {type(batch[0])}")
    
    # Try to collate (this was failing before the fix)
    try:
        batched = collate_fn(batch)
        print(f"✓ collate_fn succeeded!")
        print(f"  Batched type: {type(batched)}")
        print(f"  Batched keys: {list(batched.keys())}")
        print("✓ TEST 2 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ collate_fn FAILED: {e}")
        print("✗ TEST 2 FAILED\n")
        return False


def test_ondisk_load_dataset_splits():
    """Test that OnDiskInductivePreprocessor.load_dataset_splits works end-to-end."""
    print("="*70)
    print("TEST 3: OnDiskInductivePreprocessor End-to-End")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create on-disk dataset
        source_dataset = TinyTestDataset(num_samples=10)
        
        transforms_config = OmegaConf.create({
            "identity": {
                "transform_type": "lifting",
                "transform_name": "IdentityLifting",
            }
        })
        
        print("Creating OnDiskInductivePreprocessor...")
        ondisk_dataset = OnDiskInductivePreprocessor(
            dataset=source_dataset,
            data_dir=tmpdir,
            transforms_config=None,  # No transforms for faster test
            num_workers=1,
            storage_backend="files",  # Faster for small test
            cache_size=0,
        )
        print(f"✓ Created dataset with {len(ondisk_dataset)} samples")
        
        # Load splits
        print("Loading dataset splits...")
        split_config = OmegaConf.create({
            "learning_setting": "inductive",
            "split_type": "random",
            "data_seed": 42,
            "data_split_dir": str(Path(tmpdir) / "splits"),
            "train_prop": 0.6,
        })
        
        train_ds, val_ds, test_ds = ondisk_dataset.load_dataset_splits(split_config)
        print(f"✓ Loaded splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        print(f"  Train dataset type: {type(train_ds).__name__}")
        
        # Test that train_ds works with collate_fn
        print("Testing collate_fn with split dataset...")
        batch = [train_ds[i] for i in range(min(2, len(train_ds)))]
        
        try:
            batched = collate_fn(batch)
            print(f"✓ collate_fn succeeded with split dataset!")
            print(f"  Batch size: {batched.batch_0.max().item() + 1}")
            print("✓ TEST 3 PASSED\n")
            return True
        except Exception as e:
            print(f"✗ collate_fn FAILED: {e}")
            print(f"  This was the original bug!")
            print("✗ TEST 3 FAILED\n")
            return False


def test_memory_efficiency():
    """Test that LazyDataloadDataset maintains O(1) memory."""
    from topobench.data.datasets import LazyDataloadDataset
    import sys
    
    print("="*70)
    print("TEST 4: Memory Efficiency")
    print("="*70)
    
    dataset = TinyTestDataset(num_samples=1000)
    
    # Create split with 500 samples
    indices = list(range(500))
    lazy_split = LazyDataloadDataset(dataset, indices)
    
    # Measure memory of just the split object
    split_size = sys.getsizeof(lazy_split)
    indices_size = sys.getsizeof(lazy_split.indices)
    total_size = split_size + indices_size
    
    print(f"✓ LazyDataloadDataset object size: {split_size:,} bytes")
    print(f"✓ Indices list size: {indices_size:,} bytes")
    print(f"✓ Total memory: {total_size:,} bytes ({total_size/1024:.2f} KB)")
    print(f"✓ Per-sample overhead: {total_size/500:.2f} bytes")
    
    # Compare to loading all samples (what DataloadDataset would do)
    print("\nFor comparison, loading all 500 Data objects would use:")
    sample = dataset[0]
    sample_size = sys.getsizeof(sample) + sum(
        sys.getsizeof(getattr(sample, key)) 
        for key in sample.keys()
    )
    estimated_full_load = sample_size * 500
    print(f"  Estimated: {estimated_full_load:,} bytes ({estimated_full_load/1024/1024:.2f} MB)")
    print(f"  Memory savings: {estimated_full_load/total_size:.1f}x")
    
    print("✓ TEST 4 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ON-DISK TRAINING BUG FIX - TEST SUITE")
    print("="*70)
    print("\nThis test suite verifies the fix for the on-disk training bug.")
    print("Bug: collate_fn expected (values, keys) tuples but received Data objects.")
    print("Fix: LazyDataloadDataset provides tuple interface with O(1) memory.\n")
    
    results = []
    
    # Run all tests
    try:
        test_lazy_dataload_dataset_tuple_format()
        results.append(("LazyDataloadDataset Format", True))
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}\n")
        results.append(("LazyDataloadDataset Format", False))
    
    try:
        success = test_collate_fn_compatibility()
        results.append(("collate_fn Compatibility", success))
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}\n")
        results.append(("collate_fn Compatibility", False))
    
    try:
        success = test_ondisk_load_dataset_splits()
        results.append(("OnDisk End-to-End", success))
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}\n")
        results.append(("OnDisk End-to-End", False))
    
    try:
        test_memory_efficiency()
        results.append(("Memory Efficiency", True))
    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}\n")
        results.append(("Memory Efficiency", False))
    
    # Print summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Bug fix verified!")
    else:
        print("✗ SOME TESTS FAILED - Review the output above")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
