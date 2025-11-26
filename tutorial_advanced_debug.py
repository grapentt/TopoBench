"""Debug script for tutorial_ondisk_inductive_advanced.ipynb

Tests DAG caching functionality with a smaller dataset.
"""
import warnings
warnings.filterwarnings('ignore')

import time
import shutil
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from omegaconf import OmegaConf

# Direct import to avoid dataloader dependency issues
import sys
sys.path.insert(0, '/home/tgrapentin/personal/tdl/Topo2/TopoBench')
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductivePreprocessor


class QuickStartDataset(InMemoryDataset):
    """Lightweight synthetic graph dataset for tutorial."""
    
    def __init__(self, root, num_samples=1000, num_nodes=30, num_features=16, seed=42):
        self._num_samples = num_samples
        self._num_nodes = num_nodes
        self._num_features = num_features
        self._seed = seed
        super().__init__(root)
        
        out = fs.torch_load(self.processed_paths[0])
        if len(out) == 4:
            data, self.slices, self.sizes, data_cls = out
            self.data = data_cls.from_dict(data) if isinstance(data, dict) else data
        else:
            data, self.slices, self.sizes = out
            self.data = data
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return "data.pt"
    
    def download(self):
        pass
    
    def process(self):
        """Generate synthetic graphs using Watts-Strogatz model."""
        data_list = []
        
        for i in range(self._num_samples):
            G = nx.watts_strogatz_graph(
                n=self._num_nodes, k=6, p=0.3, seed=self._seed + i
            )
            
            edges = list(G.edges())
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            
            x = torch.randn(self._num_nodes, self._num_features)
            y = torch.tensor([i % 3])
            
            data_list.append(Data(
                x=x, edge_index=edge_index, y=y, num_nodes=self._num_nodes
            ))
        
        self.data, self.slices = self.collate(data_list)
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0]
        )


def main():
    """Main test function."""
    print("=" * 70)
    print("DAG Caching Tutorial - Debug Script")
    print("=" * 70)
    
    # Clean up old data
    for path in ["./data/tutorial_advanced_source", "./data/dag_demo"]:
        if Path(path).exists():
            shutil.rmtree(path)
            print(f"✓ Cleaned up {path}")
    
    # Create dataset with smaller size for faster testing
    print("\n1. Creating dataset...")
    dataset = QuickStartDataset(
        root="./data/tutorial_advanced_source",
        num_samples=100,  # Small for fast testing
        num_nodes=20,
        num_features=16,
        seed=42
    )
    print(f"✓ Dataset ready: {len(dataset)} graphs")
    
    # Experiment 1: Baseline (Clique Lifting only)
    print("\n" + "=" * 70)
    print("[Experiment 1] Baseline: SimplicialCliqueLifting")
    print("=" * 70)
    
    config1 = OmegaConf.create({
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2
        }
    })
    
    start = time.time()
    preprocessor1 = OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir="./data/dag_demo",
        transforms_config=config1,
        storage_backend="files",
        num_workers=2,  # Reduced for cleaner output
        force_reload=True  # Force fresh processing
    )
    time1 = time.time() - start
    
    print(f"\n✓ Baseline completed in {time1:.2f}s")
    print(f"  Transform chain: {len(preprocessor1.transform_chain)} transform(s)")
    for i, entry in enumerate(preprocessor1.transform_chain):
        print(f"    [{i}] {entry['transform_class']} - cached={entry['cached']}")
    
    # Verify data is accessible
    sample = preprocessor1[0]
    print(f"  Sample check: x_0={sample.x_0.shape}, x_1={sample.x_1.shape}, x_2={sample.x_2.shape}")
    
    # Experiment 2: Add ProjectionSum (should reuse clique lifting!)
    print("\n" + "=" * 70)
    print("[Experiment 2] Add ProjectionSum (DAG cache reuse)")
    print("=" * 70)
    
    config2 = OmegaConf.create({
        "clique_lifting": {  # Same config = should be cached!
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2
        },
        "projection": {  # NEW transform
            "transform_type": "feature",
            "transform_name": "ProjectionSum"
        }
    })
    
    start = time.time()
    preprocessor2 = OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir="./data/dag_demo",  # Same directory!
        transforms_config=config2,
        storage_backend="files",
        num_workers=2
    )
    time2 = time.time() - start
    
    print(f"\n✓ With DAG cache completed in {time2:.2f}s")
    print(f"  Transform chain: {len(preprocessor2.transform_chain)} transform(s)")
    for i, entry in enumerate(preprocessor2.transform_chain):
        print(f"    [{i}] {entry['transform_class']} - cached={entry['cached']}")
    
    # Verify data is accessible
    sample = preprocessor2[0]
    print(f"  Sample check: x_0={sample.x_0.shape}, x_1={sample.x_1.shape}, x_2={sample.x_2.shape}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Baseline (lifting only):     {time1:.2f}s")
    print(f"Add transform (with cache):  {time2:.2f}s")
    
    if time2 < time1:
        speedup = time1 / time2
        print(f"✅ DAG cache working! Speedup: {speedup:.2f}×")
    else:
        print(f"⚠️  Expected time2 < time1")
        print(f"   This might indicate the cache is not being reused properly")
    
    # Check cache status
    print("\nCache verification:")
    if preprocessor2.transform_chain[0]['cached']:
        print("  ✅ First transform (clique_lifting) marked as cached")
    else:
        print("  ❌ First transform NOT cached - BUG!")
    
    if not preprocessor2.transform_chain[1]['cached']:
        print("  ✅ Second transform (projection) was newly processed")
    else:
        print("  ⚠️  Second transform marked as cached (unexpected)")
    
    # Check directories
    print("\nCache directories:")
    for i, entry in enumerate(preprocessor2.transform_chain):
        output_dir = Path(entry['output_dir'])
        exists = "✓" if output_dir.exists() else "✗"
        print(f"  {exists} [{i}] {output_dir}")
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
