"""Manual test to verify DAG caching behavior."""
import tempfile
import time
from pathlib import Path

import torch
from torch_geometric.data import Data
from omegaconf import OmegaConf

from topobench.data.preprocessor.ondisk_inductive import OnDiskInductivePreprocessor


class SimpleDataset(torch.utils.data.Dataset):
    """Simple test dataset."""
    
    def __init__(self, num_samples=50):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(42 + idx)
        return Data(
            x=torch.randn(10, 8),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            y=torch.tensor([idx % 3]),
        )


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        source = SimpleDataset(num_samples=50)
        
        print("=" * 60)
        print("Test 1: Process with lifting only")
        print("=" * 60)
        
        config1 = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2,
            },
        })
        
        start1 = time.time()
        dataset1 = OnDiskInductivePreprocessor(
            dataset=source,
            data_dir=data_dir,
            transforms_config=config1,
            num_workers=1,
            storage_backend="files",
        )
        time1 = time.time() - start1
        
        print(f"\n✓ Time: {time1:.2f}s")
        print(f"Transform chain: {len(dataset1.transform_chain)} transform(s)")
        for i, entry in enumerate(dataset1.transform_chain):
            print(f"  [{i}] {entry['transform_class']} - cached={entry['cached']}")
        
        print("\n" + "=" * 60)
        print("Test 2: Add ProjectionSum (should reuse lifting!)")
        print("=" * 60)
        
        config2 = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2,
            },
            "projection": {
                "transform_type": "feature",
                "transform_name": "ProjectionSum",
            },
        })
        
        start2 = time.time()
        dataset2 = OnDiskInductivePreprocessor(
            dataset=source,
            data_dir=data_dir,  # Same directory!
            transforms_config=config2,
            num_workers=1,
            storage_backend="files",
        )
        time2 = time.time() - start2
        
        print(f"\n✓ Time: {time2:.2f}s")
        print(f"Transform chain: {len(dataset2.transform_chain)} transform(s)")
        for i, entry in enumerate(dataset2.transform_chain):
            print(f"  [{i}] {entry['transform_class']} - cached={entry['cached']}")
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Baseline (lifting only):     {time1:.2f}s")
        print(f"Add transform (with cache):  {time2:.2f}s")
        
        if time2 < time1:
            print(f"✅ DAG cache working! ({time1/time2:.2f}× faster)")
        else:
            print(f"⚠️  Expected time2 < time1, got time2={time2:.2f}s >= time1={time1:.2f}s")
        
        # Verify first transform was reused
        if dataset2.transform_chain[0]['cached']:
            print("✅ First transform marked as cached")
        else:
            print("❌ First transform NOT marked as cached (BUG!)")


if __name__ == "__main__":
    main()
