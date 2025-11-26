"""Full test of tutorial_ondisk_inductive_advanced workflow."""
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


def test_dag_caching():
    """Test DAG caching demonstration."""
    print("=" * 70)
    print("TEST 1: DAG Caching Demonstration")
    print("=" * 70)
    
    # Clean up
    for path in ["./data/tutorial_advanced_source", "./data/dag_demo"]:
        if Path(path).exists():
            shutil.rmtree(path)
    
    dataset = QuickStartDataset(
        root="./data/tutorial_advanced_source",
        num_samples=200,
        num_nodes=20,
        seed=42
    )
    print(f"✓ Dataset: {len(dataset)} graphs\n")
    
    # Experiment 1
    config1 = OmegaConf.create({
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2
        }
    })
    
    print("[Experiment 1] Baseline")
    start = time.time()
    p1 = OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir="./data/dag_demo",
        transforms_config=config1,
        storage_backend="files",
        num_workers=2,
        force_reload=True
    )
    time1 = time.time() - start
    print(f"✓ Time: {time1:.2f}s\n")
    
    # Experiment 2
    config2 = OmegaConf.create({
        "clique_lifting": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2
        },
        "projection": {
            "transform_type": "feature",
            "transform_name": "ProjectionSum"
        }
    })
    
    print("[Experiment 2] Add transform")
    start = time.time()
    p2 = OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir="./data/dag_demo",
        transforms_config=config2,
        storage_backend="files",
        num_workers=2
    )
    time2 = time.time() - start
    
    speedup = time1 / time2
    print(f"✓ Time: {time2:.2f}s (Speedup: {speedup:.2f}×)\n")
    
    assert speedup > 1.5, f"Expected speedup > 1.5×, got {speedup:.2f}×"
    assert p2.transform_chain[0]['cached'], "First transform should be cached"
    print("✅ DAG caching test PASSED\n")
    
    return time1, time2, speedup


def test_storage_backends():
    """Test storage backend comparison."""
    print("=" * 70)
    print("TEST 2: Storage Backends Comparison")
    print("=" * 70)
    
    # Clean up
    for path in ["./data/small_source", "./data/compare_files", "./data/compare_mmap"]:
        if Path(path).exists():
            shutil.rmtree(path)
    
    dataset = QuickStartDataset(
        root="./data/small_source",
        num_samples=100,
        num_nodes=20,
        seed=100
    )
    
    config = OmegaConf.create({
        "clique": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2
        }
    })
    
    # Files backend
    print("[Files Backend]")
    start = time.time()
    p_files = OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir="./data/compare_files",
        transforms_config=config,
        storage_backend="files",
        num_workers=2
    )
    time_files = time.time() - start
    print(f"✓ Time: {time_files:.2f}s\n")
    
    # Mmap backend
    print("[Mmap Backend]")
    start = time.time()
    p_mmap = OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir="./data/compare_mmap",
        transforms_config=config,
        storage_backend="mmap",
        compression="lz4",
        num_workers=1
    )
    time_mmap = time.time() - start
    print(f"✓ Time: {time_mmap:.2f}s\n")
    
    # Verify both can load data
    sample_files = p_files[0]
    sample_mmap = p_mmap[0]
    assert hasattr(sample_files, 'x_0'), "Files backend data check failed"
    assert hasattr(sample_mmap, 'x_0'), "Mmap backend data check failed"
    
    print("✅ Storage backends test PASSED\n")
    
    return time_files, time_mmap


def test_parallel_processing():
    """Test parallel processing demonstration."""
    print("=" * 70)
    print("TEST 3: Parallel Processing")
    print("=" * 70)
    
    # Clean up
    for path in ["./data/test_source"] + [f"./data/parallel_{w}" for w in [1, 2]]:
        if Path(path).exists():
            shutil.rmtree(path)
    
    dataset = QuickStartDataset(
        root="./data/test_source",
        num_samples=150,
        num_nodes=20,
        seed=200
    )
    
    config = OmegaConf.create({
        "clique": {
            "transform_type": "lifting",
            "transform_name": "SimplicialCliqueLifting",
            "complex_dim": 2
        }
    })
    
    times = {}
    for workers in [1, 2]:
        print(f"[{workers} worker(s)]")
        start = time.time()
        p = OnDiskInductivePreprocessor(
            dataset=dataset,
            data_dir=f"./data/parallel_{workers}",
            transforms_config=config,
            storage_backend="files",
            num_workers=workers,
            force_reload=True
        )
        times[workers] = time.time() - start
        print(f"✓ Time: {times[workers]:.2f}s\n")
    
    speedup = times[1] / times[2]
    print(f"Parallel speedup (2 workers): {speedup:.2f}×")
    # Note: With small datasets, overhead reduces parallel gains
    assert speedup >= 1.0, f"Expected speedup >= 1.0×, got {speedup:.2f}×"
    print("✅ Parallel processing test PASSED\n")
    
    return times


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FULL TUTORIAL TEST SUITE")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Test 1: DAG Caching
    time1, time2, speedup = test_dag_caching()
    results['dag'] = {'time1': time1, 'time2': time2, 'speedup': speedup}
    
    # Test 2: Storage Backends  
    time_files, time_mmap = test_storage_backends()
    results['storage'] = {'files': time_files, 'mmap': time_mmap}
    
    # Test 3: Parallel Processing
    times = test_parallel_processing()
    results['parallel'] = times
    
    # Final Summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"✅ DAG Caching:       {results['dag']['speedup']:.2f}× speedup")
    print(f"✅ Storage Backends:  Files={results['storage']['files']:.2f}s, Mmap={results['storage']['mmap']:.2f}s")
    print(f"✅ Parallel (2 workers): {times[1] / times[2]:.2f}× speedup")
    print("\n🎉 All tests PASSED! Tutorial is working correctly.\n")


if __name__ == "__main__":
    main()
