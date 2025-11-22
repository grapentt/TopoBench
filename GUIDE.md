# TopoBench On-Disk Dataset Guide

**User Guide for Large-Scale Inductive and Transductive Learning**

---

## Table of Contents

1. [Introduction](#introduction)
2. [When to Use On-Disk](#when-to-use-on-disk)
3. [Quick Start: Inductive](#quick-start-inductive)
4. [Quick Start: Transductive](#quick-start-transductive)
5. [Memory/Disk Configuration](#memorydisk-configuration)
6. [Adding Custom Datasets](#adding-custom-datasets)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)
9. [Performance Characteristics](#performance-characteristics)

---

## Introduction

TopoBench's on-disk infrastructure enables training on datasets **larger than available RAM** by processing data sequentially and storing it on disk, maintaining constant memory usage regardless of dataset size.

### The Problem: Memory Exhaustion

Traditional in-memory preprocessing loads ALL topological structures into RAM:
- **Edges**: O(N × D) structures
- **Triangles**: O(N × D²) structures  
- **4-cliques**: O(N × D³) structures

For large graphs or many graphs, this causes **Out-Of-Memory (OOM)** errors.

### The Solution: On-Disk Processing

- **Inductive (many graphs)**: Process graphs one-by-one, save to disk immediately
- **Transductive (one large graph)**: Index structures to disk, query on-demand
- **Result**: **Constant O(1) memory** per sample

---

## When to Use On-Disk

### Use On-Disk When:

✅ **Dataset has many graphs** (inductive)
- More than 1000 graphs
- Each graph > 50 nodes
- High degree (> 10) → many triangles

✅ **Graph is very large** (transductive)
- More than 10,000 nodes
- High degree → millions of triangles
- Available RAM < 2× structures memory

✅ **Limited RAM**
- Less than 8GB available
- Other processes using memory
- Training on laptop/resource-constrained environment

### Use In-Memory When:

✅ **Small datasets** (< 500 graphs, < 5000 nodes)
✅ **Plenty of RAM** (> 16GB available)
✅ **Speed critical** (in-memory is ~1.5-2x faster)

---

## Quick Start: Inductive

### Step 1: Load Your Dataset

```python
from omegaconf import OmegaConf
from topobench.data.loaders.graph import TUDatasetLoader

# Configure loader
loader_config = OmegaConf.create({
    "data_dir": "./data/",
    "data_name": "ENZYMES",
    "data_type": "TUDataset"
})

# Load dataset
loader = TUDatasetLoader(loader_config)
dataset, data_dir = loader.load()
```

### Step 2: Apply On-Disk Preprocessing

```python
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductiveDataset

# Configure lifting
transforms_config = OmegaConf.create({
    "clique_lifting": {
        "transform_type": "lifting",
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 2  # Include triangles
    }
})

# Create on-disk dataset (constant memory!)
ondisk_dataset = OnDiskInductiveDataset(
    dataset=dataset,
    data_dir="./data/enzymes_ondisk",
    transforms_config=transforms_config,
    force_reload=False  # Use cache if exists
)

print(f"Processed {len(ondisk_dataset)} graphs")
```

### Step 3: Create Splits

```python
from omegaconf import DictConfig

# Configure splits
split_config = DictConfig({
    "learning_setting": "inductive",
    "split_type": "random",
    "data_seed": 0,
    "train_prop": 0.6,
    "data_split_dir": "./data/enzymes_ondisk/splits"
})

# Load splits (compatible with existing utilities!)
train_dataset, val_dataset, test_dataset = ondisk_dataset.load_dataset_splits(split_config)
```

### Step 4: Train Your Model

```python
from topobench.dataloader import TBDataloader
import pytorch_lightning as pl
from topobench.model import TBModel
# ... (import your model components)

# Create dataloader
datamodule = TBDataloader(train_dataset, val_dataset, test_dataset, batch_size=32)

# Setup model (same as tutorial_model.ipynb)
model = TBModel(
    backbone=backbone_wrapper,
    readout=readout,
    loss=loss,
    feature_encoder=feature_encoder,
    evaluator=evaluator,
    optimizer=optimizer
)

# Train (exactly like in-memory!)
trainer = pl.Trainer(max_epochs=50, accelerator="cpu")
trainer.fit(model, datamodule)
```

---

## Quick Start: Transductive

### Step 1: Load Your Graph

```python
from topobench.data.loaders.graph import CoraLoader  # or your loader

loader_config = OmegaConf.create({
    "data_dir": "./data/",
    "data_name": "Cora"
})

loader = CoraLoader(loader_config)
dataset, data_dir = loader.load()
graph_data = dataset[0]  # Single graph for transductive
```

### Step 2: Create On-Disk Transductive Dataset

```python
from topobench.data.preprocessor.ondisk_transductive import OnDiskTransductiveDataset

# Create dataset with structure indexing
ondisk_dataset = OnDiskTransductiveDataset(
    graph_data=graph_data,
    data_dir="./data/cora_ondisk",
    max_structure_size=3,  # Index triangles
    force_rebuild=False
)

# Build index (one-time operation)
ondisk_dataset.build_index()
print(f"Indexed {ondisk_dataset.num_structures} structures")
```

### Step 3: Query Structures During Training

```python
# Query structures for a batch of nodes
batch_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example batch
structures = ondisk_dataset.query_batch(batch_nodes, fully_contained=True)

print(f"Found {len(structures)} structures in batch")

# Each structure is (structure_id, node_list)
for struct_id, nodes in structures[:3]:
    print(f"Structure {struct_id}: nodes {nodes}")
```

### Step 4: Training Integration

**Note**: Full training integration is under development. Current approach:

```python
# Option 1: Use query_batch in custom training loop
for epoch in range(num_epochs):
    for batch_nodes in node_sampler:
        # Query structures for this batch
        structures = ondisk_dataset.query_batch(batch_nodes, fully_contained=True)
        
        # Construct batch with structures
        batch = construct_batch(batch_nodes, structures, graph_data)
        
        # Train on batch
        loss = model(batch)
        loss.backward()
        optimizer.step()

# Option 2: Use provided dataloader wrapper (when available)
# from topobench.dataloader import OnDiskTransductiveDataloader
# dataloader = OnDiskTransductiveDataloader(ondisk_dataset, ...)
```

---

## Memory/Disk Configuration

### Calculating Dataset Size

**Formula for triangles** (complex_dim=2):
```
Memory (GB) ≈ num_graphs × nodes_per_graph × degree² × 12 bytes / 1e9
```

**Example**:
- 5000 graphs
- 80 nodes each
- Degree 15
- Triangles: ~5000 × 80 × 15² × 12 / 1e9 ≈ 5.4 GB

### Choosing Parameters for Your RAM

**If you have 4GB RAM available**:
```python
# Target: ~3GB (leave headroom)
# Using formula: 3e9 = N × 80 × 225 × 12
# N = 3e9 / (80 × 225 × 12) ≈ 1,388 graphs

loader_config = OmegaConf.create({
    "num_graphs": 1300,  # Safe for 4GB
    "nodes_per_graph": 80,
    "degree": 15,
    # ...
})
```

### Disk Requirements

- **Inductive**: ~100-500MB per 1000 graphs (depends on graph size)
- **Transductive**: ~50-200MB for index (depends on num_structures)
- **Rule of thumb**: Disk ≈ 2-3× in-memory size (structures + overhead)

---

## Adding Custom Datasets

### For Inductive Datasets

#### Step 1: Create Dataset Class

```python
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from omegaconf import DictConfig
import os.path as osp

class MyCustomDataset(InMemoryDataset):
    """Your custom inductive dataset."""
    
    def __init__(self, root, name, parameters: DictConfig):
        self.name = name
        self.parameters = parameters
        super().__init__(root)
        
        # Load processed data
        out = fs.torch_load(self.processed_paths[0])
        if len(out) == 4:
            data, self.slices, self.sizes, data_cls = out
            self.data = data_cls.from_dict(data) if isinstance(data, dict) else data
        else:
            data, self.slices, self.sizes = out
            self.data = data
    
    @property
    def raw_file_names(self):
        # Return list of raw files (or empty if generating)
        return []
    
    @property
    def processed_file_names(self):
        return "data.pt"
    
    def download(self):
        # Download your data if needed
        pass
    
    def process(self):
        """Load/generate your graphs."""
        data_list = []
        
        # Your logic to create graphs
        for i in range(self.parameters.num_graphs):
            # Create PyG Data objects
            data = Data(x=..., edge_index=..., y=...)
            data_list.append(data)
        
        # Collate and save
        self.data, self.slices = self.collate(data_list)
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0]
        )
```

#### Step 2: Create Loader

```python
from topobench.data.loaders.base import AbstractLoader
from pathlib import Path

class MyCustomLoader(AbstractLoader):
    """Loader for your custom dataset."""
    
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
    
    def load_dataset(self):
        dataset = MyCustomDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters
        )
        return dataset
```

#### Step 3: Use with On-Disk

```python
# Load your dataset
loader = MyCustomLoader(config)
dataset, data_dir = loader.load()

# Process on-disk
ondisk_dataset = OnDiskInductiveDataset(
    dataset=dataset,
    data_dir="./my_dataset_ondisk",
    transforms_config=transforms_config
)

# Train as usual!
```

### For Transductive Datasets

For transductive, your dataset should return a **single graph** with train/val/test masks:

```python
class MyTransductiveDataset(InMemoryDataset):
    def process(self):
        # Create single large graph
        data = Data(
            x=..., 
            edge_index=..., 
            y=...,
            train_mask=...,
            val_mask=...,
            test_mask=...
        )
        
        data_list = [data]  # Single graph in list
        self.data, self.slices = self.collate(data_list)
        # ...
```

Then use `OnDiskTransductiveDataset` as shown in Quick Start.

---

## Troubleshooting

### Issue: "Dataset still causes OOM"

**Possible causes**:
1. **Not using on-disk**: Make sure you're using `OnDiskInductiveDataset`, not `PreProcessor`
2. **Dataset too large even for on-disk**: Check disk space, reduce batch size
3. **Model too large**: Reduce model parameters, hidden dimensions

**Solution**:
```python
# Verify you're using on-disk
from topobench.data.preprocessor.ondisk_inductive import OnDiskInductiveDataset

ondisk_dataset = OnDiskInductiveDataset(...)  # Not PreProcessor!

# Reduce batch size if model is large
datamodule = TBDataloader(..., batch_size=16)  # Smaller batch
```

### Issue: "Processing is very slow"

**Expected**: On-disk is ~1.5-2x slower than in-memory (disk I/O overhead)

**If extremely slow**:
1. **Check disk type**: SSD much faster than HDD
2. **Check disk space**: Full disk causes slowdowns
3. **Reduce structure size**: Use `complex_dim=2` (triangles) instead of higher

**Optimization**:
```python
# Use SSD if available
data_dir = "/path/to/ssd/ondisk_data"  # Not HDD!

# Cache preprocessing results
ondisk_dataset = OnDiskInductiveDataset(
    ...,
    force_reload=False  # Reuse cache
)
```

### Issue: "Correctness: Results differ from in-memory"

**For transductive**, verify query correctness:
```python
from topobench.data.structure_query import verify_query_correctness

# Validate against in-memory baseline
batch_nodes = [0, 1, 2, 3, 4]
results = verify_query_correctness(ondisk_dataset.query_engine, batch_nodes)

if results["correct"]:
    print("✓ Query results correct!")
else:
    print(f"❌ Missing {len(results['missing'])} structures")
    print(f"❌ Extra {len(results['extra'])} structures")
```

**For inductive**, compare with `PreProcessor` on small dataset:
```python
# Small test dataset
small_dataset = dataset[:10]

# Process with both
preprocessor = PreProcessor(small_dataset, dir1, transforms)
ondisk = OnDiskInductiveDataset(small_dataset, dir2, transforms)

# Compare outputs (should be identical or mathematically equivalent)
for i in range(len(small_dataset)):
    data1 = preprocessor[i]
    data2 = ondisk[i]
    # Verify structures match
```

### Issue: "File not found errors"

**Cause**: Metadata or sample files corrupted/deleted

**Solution**:
```python
# Force rebuild
ondisk_dataset = OnDiskInductiveDataset(
    ...,
    force_reload=True  # Recreate all files
)
```

### Issue: "Transform not working"

**Check**:
1. Transform config format matches TopoBench pattern
2. Transform is compatible with sequential processing
3. Transform parameters are serializable

**Example correct config**:
```python
transforms_config = OmegaConf.create({
    "clique_lifting": {
        "transform_type": "lifting",
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 2
    }
})
```

---

## API Reference

### OnDiskInductiveDataset

```python
class OnDiskInductiveDataset(torch.utils.data.Dataset):
    """Sequential disk-backed dataset for inductive learning.
    
    Args:
        dataset: Source dataset (PyG or PyTorch)
        data_dir: Directory for processed samples
        transforms_config: Transform configuration (DictConfig)
        force_reload: Rebuild even if cache exists (default: False)
    
    Methods:
        __len__(): Number of samples
        __getitem__(idx): Load sample from disk
        load_dataset_splits(split_params): Create train/val/test splits
    
    Attributes:
        num_samples: Total samples
        processed_dir: Path to processed directory
        metadata_path: Path to metadata file
    """
```

**Usage**:
```python
dataset = OnDiskInductiveDataset(
    dataset=source,
    data_dir="./processed",
    transforms_config=config,
    force_reload=False
)

# Load sample
data = dataset[0]

# Create splits
train, val, test = dataset.load_dataset_splits(split_config)
```

### OnDiskTransductiveDataset

```python
class OnDiskTransductiveDataset(torch.utils.data.Dataset):
    """On-disk dataset for transductive learning with structure indexing.
    
    Args:
        graph_data: Single graph (PyG Data)
        data_dir: Directory for structure index
        max_structure_size: Max clique size (default: 3 for triangles)
        force_rebuild: Rebuild index if exists (default: False)
    
    Methods:
        build_index(): Build/load structure index
        query_batch(node_ids, fully_contained): Query structures for batch
        get_subgraph(node_ids): Extract subgraph with structures
        get_stats(): Get index statistics
        close(): Cleanup resources
    
    Attributes:
        num_nodes: Number of nodes in graph
        num_structures: Total structures indexed
        query_engine: StructureQueryEngine instance
    """
```

**Usage**:
```python
dataset = OnDiskTransductiveDataset(
    graph_data=graph,
    data_dir="./index",
    max_structure_size=3
)

# Build index (one-time)
dataset.build_index()

# Query for batch
structures = dataset.query_batch([0,1,2,3,4], fully_contained=True)

# Cleanup
dataset.close()
```

### StructureQueryEngine

```python
class StructureQueryEngine:
    """High-level interface for structure queries.
    
    Args:
        graph: NetworkX graph
        index_dir: Directory for index
        max_structure_size: Max clique size
        force_rebuild: Rebuild if exists (default: False)
    
    Methods:
        open(): Open backend connection
        close(): Close backend
        build_index(): Build/load index
        query_batch(node_ids, fully_contained): Query structures
        get_stats(): Get index stats
    
    Context Manager:
        with StructureQueryEngine(...) as engine:
            engine.build_index()
            structures = engine.query_batch(...)
    """
```

---

## Performance Characteristics

### Memory Usage

| Approach | Memory Usage | Scales With |
|----------|--------------|-------------|
| In-Memory | O(N × D^k) | Dataset size |
| On-Disk Inductive | O(1) per sample | Single sample |
| On-Disk Transductive | O(B × D^k) | Batch size |

Where:
- N = number of nodes/graphs
- D = average degree
- k = structure dimension (3 for triangles)
- B = batch size

### Speed Comparison

| Operation | In-Memory | On-Disk | Overhead |
|-----------|-----------|---------|----------|
| Preprocessing | 1.0x (baseline) | 1.5-2.0x | Disk I/O |
| Training | 1.0x | 1.1-1.3x | Disk loading |
| Total | 1.0x | 1.3-1.8x | Acceptable |

### Disk Usage

- **Inductive**: ~100-500MB per 1000 graphs
- **Transductive**: ~50-200MB for structure index
- **Rule**: Disk ≈ 2-3× in-memory size

### When Is Overhead Worth It?

**Overhead acceptable when**:
- Dataset won't fit in memory (no choice!)
- RAM limited but disk plentiful
- Preprocessing once, using many times (cache)

**Overhead NOT worth it when**:
- Small dataset fits easily in memory
- SSD/disk very slow (e.g., network drive)
- Speed absolutely critical (real-time inference)

---

## Transform Support

### Validated Transforms ✅

On-disk datasets **fully support** TopoBench transforms (liftings). All tested transforms produce **identical results** to in-memory `PreProcessor`.

**Tested & Validated**:
- ✅ **SimplicialCliqueLifting** (complex_dim: 1, 2, 3) - Triangles, tetrahedra
- ✅ **HypergraphKHopLifting** (k_value: 2+) - K-hop neighborhoods

**Usage**:
```python
from topobench.data.preprocessor import create_preprocessor
from omegaconf import OmegaConf

transforms_config = OmegaConf.create({
    "clique_lifting": {
        "transform_type": "lifting",
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 2  # Include triangles
    }
})

# Works with on-disk!
preprocessor = create_preprocessor(
    dataset=dataset,
    data_dir="./processed",
    transforms_config=transforms_config,
    mode="ondisk"  # or "auto"
)
```

**Key Features**:
- **Correctness**: Produces same structures as in-memory
- **Caching**: Transform results cached on disk (parameter-based hashing)
- **Memory**: O(1) per sample during processing

**Validation**: 10 integration tests validate correctness on MUTAG & ENZYMES datasets.

**For Details**: See `TRANSFORM_SUPPORT.md` for comprehensive transform documentation.

---

## Best Practices

1. **Use SSD not HDD** - 5-10x faster disk I/O
2. **Cache preprocessing** - Set `force_reload=False`, reuse results
3. **Start small** - Test on small dataset first, then scale up
4. **Monitor disk space** - Ensure 2-3× dataset size available
5. **Profile memory** - Use memory tracking to verify O(1) usage
6. **Validate correctness** - Use `verify_query_correctness` for transductive

---

## Examples

See tutorial notebooks for complete examples:
- `tutorials/tutorial_ondisk_inductive.ipynb` - Full inductive workflow
- `tutorials/tutorial_ondisk_transductive.ipynb` - Transductive workflow

See validation scripts for demonstrations:
- `validation_1_inmemory_inductive_fails.py` - In-memory OOM
- `validation_2_ondisk_inductive_works.py` - On-disk success
- `validation_3_inmemory_transductive_fails.py` - Transductive OOM
- `validation_4_ondisk_transductive_works.py` - Transductive success

---

## Support

For issues, questions, or contributions:
1. Check troubleshooting section above
2. Review tutorial notebooks
3. See GOAL.md for implementation details
4. Open GitHub issue with clear description

---

**Happy large-scale training!** 🚀
