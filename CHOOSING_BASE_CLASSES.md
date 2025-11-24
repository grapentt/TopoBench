# Choosing the Right Base Class

## Quick Decision Tree

```
Do you have pre-saved sample files?
├─ YES → Use FileBasedInductiveDataset
└─ NO → Is computing samples cheaper than storing all?
    ├─ YES → Use OnDemandInductiveDataset
    └─ NO → Save samples first, then use FileBasedInductiveDataset
```

## The Three Base Classes

### 1. BaseOnDiskInductiveDataset (Abstract Base)

**Don't use directly** - inherit from FileBasedInductiveDataset or OnDemandInductiveDataset instead.

**What it provides:**
- Lightweight pickling (< 1-10KB)
- O(1) memory usage
- Automatic caching
- Parallel DataLoader support

---

### 2. FileBasedInductiveDataset

**Use when:** Samples are already saved as files

```python
class USCountyDemosOnDiskDataset(FileBasedInductiveDataset):
    def __init__(self, root, ...):
        # Initialize, then call super with directory containing files
        super().__init__(
            root=self.processed_dir,  # Directory with sample_*.pt files
            file_pattern="data_*.pt",
            cache_samples=True
        )
    
    def _load_file(self, file_path: Path) -> Data:
        # Simple file loading
        return torch.load(file_path, weights_only=False)
```

**Characteristics:**
- ✅ Samples pre-saved to disk
- ✅ Loading is fast (just torch.load)
- ✅ Multiple processes access same files
- ✅ File format agnostic (PyTorch, pickle, HDF5, etc.)

**Examples:**
- Preprocessed datasets (after lifting/transformation)
- Downloaded datasets with individual sample files
- Any dataset where samples exist as files

---

### 3. OnDemandInductiveDataset

**Use when:** Computing samples on-demand is cheaper/faster than storing all

```python
class OGBNPapers100MOnDiskDataset(OnDemandInductiveDataset):
    def __init__(self, root, num_samples, k_hop=2):
        # Load memory-mapped arrays (not into RAM!)
        self.edges_mmap = np.load("edges.npy", mmap_mode='r')
        self.features_mmap = np.load("features.npy", mmap_mode='r')
        
        # Initialize with number of samples
        super().__init__(
            root=root,
            num_samples=num_samples,
            seed=42,
            cache_samples=True
        )
    
    def _generate_sample(self, idx: int, rng: torch.Generator) -> Data:
        # Compute/extract sample on-demand
        target_node = idx % self.num_nodes
        node_ids = self._extract_k_hop(target_node, self.k_hop)
        
        # Load ONLY needed data (O(k) not O(n))
        x = torch.from_numpy(self.features_mmap[node_ids].copy())
        return Data(x=x, ...)
```

**Characteristics:**
- ✅ Computes samples when accessed
- ✅ Uses deterministic seeding (reproducible)
- ✅ Can cache computed samples
- ✅ Memory-efficient for large base data

**Examples:**
1. **Synthetic generation**: Random graphs, procedural content
2. **Subgraph extraction**: k-hop neighborhoods from massive graphs
3. **Data augmentation**: Random transformations on-the-fly
4. **Memory-mapped data**: Extract portions from large arrays

**Why "Generated" for Papers100M?**

Even though Papers100M is real data (not synthetic), we "generate" the sample **object** on-demand by:
1. Extracting a k-hop subgraph from memory-mapped arrays
2. Loading only the needed node features
3. Creating a new Data object

This is faster than:
- Storing 111M pre-computed subgraphs (would be TBs!)
- Loading the full graph into RAM (44GB+)

---

## Detailed Comparison

| Aspect | FileBasedInductiveDataset | OnDemandInductiveDataset |
|--------|--------------------------|---------------------------|
| **Input** | Directory with files | Parameters for computation |
| **Init Parameters** | `root`, `file_pattern` | `root`, `num_samples`, `seed` |
| **Storage** | Files on disk | Memory-mapped or parameters |
| **Access Method** | Load from file | Compute on-demand |
| **When Faster** | Loading < computing | Computing < loading+storing |
| **Memory** | O(1) per sample | O(1) per sample |
| **Pickle Size** | ~1 KB | ~1 KB |
| **Use Cases** | Post-preprocessing, pre-saved data | Synthetic, subgraphs, large data |

---

## Real-World Examples

### Example 1: Small Dataset (< 1000 samples)

**Scenario**: ENZYMES dataset (600 graphs), apply lifting

```python
# Step 1: Preprocess with OnDiskInductivePreprocessor
preprocessor = OnDiskInductivePreprocessor(
    dataset=enzymes,
    data_dir="./preprocessed_enzymes",
    transforms_config=lifting_config,
    num_workers=4
)
# Saves: ./preprocessed_enzymes/sample_000000.pt ... sample_000599.pt

# Step 2: Load efficiently with FileBasedInductiveDataset
class PreprocessedENZYMES(FileBasedInductiveDataset):
    def _load_file(self, path):
        return torch.load(path, weights_only=False)

dataset = PreprocessedENZYMES("./preprocessed_enzymes")
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

✅ **Use FileBasedInductiveDataset** - samples already saved

---

### Example 2: Massive Single Graph (Papers100M)

**Scenario**: 111M nodes, extract k-hop neighborhoods

```python
# Don't save 111M pre-computed subgraphs!
# Instead, use OnDemandInductiveDataset

class Papers100MSubgraphs(OnDemandInductiveDataset):
    def __init__(self, root, num_samples=10000):
        # Memory-mapped arrays (not loaded into RAM)
        self.edges_mmap = np.load("papers100m_edges.npy", mmap_mode='r')
        self.feats_mmap = np.load("papers100m_feats.npy", mmap_mode='r')
        super().__init__(root, num_samples, seed=42)
    
    def _generate_sample(self, idx, rng):
        # Extract k-hop on-demand
        node_ids = self._k_hop_subgraph(idx, k=2)
        x = torch.from_numpy(self.feats_mmap[node_ids].copy())
        return Data(x=x, ...)

dataset = Papers100MSubgraphs("./cache", num_samples=10000)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

✅ **Use OnDemandInductiveDataset** - computing cheaper than storing

---

### Example 3: Synthetic Benchmark

**Scenario**: Generate 10,000 random graphs for testing

```python
class RandomGraphs(OnDemandInductiveDataset):
    def __init__(self, root, num_samples=10000):
        self.num_nodes = 50
        self.num_edges = 100
        super().__init__(root, num_samples, seed=42)
    
    def _generate_sample(self, idx, rng):
        # Generate random graph
        x = torch.randn(self.num_nodes, 8, generator=rng)
        edge_index = torch.randint(
            0, self.num_nodes, (2, self.num_edges), generator=rng
        )
        y = torch.tensor([idx % 5])
        return Data(x=x, edge_index=edge_index, y=y)

dataset = RandomGraphs("./synthetic", num_samples=10000)
```

✅ **Use OnDemandInductiveDataset** - purely synthetic

---

## Common Pitfalls

### ❌ Pitfall 1: Using FileBasedInductiveDataset for Papers100M

```python
# DON'T DO THIS!
# Trying to save 111M pre-computed subgraphs

for i in range(111_000_000):
    subgraph = extract_subgraph(i)
    torch.save(subgraph, f"sample_{i:09d}.pt")  # TBs of storage!

dataset = FileBasedInductiveDataset("./samples")  # Impractical!
```

**Why bad:** Requires TBs of storage, slow to save/load

**Solution:** Use OnDemandInductiveDataset with memory-mapping

---

### ❌ Pitfall 2: Using OnDemandInductiveDataset for Pre-Saved Data

```python
# DON'T DO THIS!
# You already have files, don't re-compute them!

class MyDataset(OnDemandInductiveDataset):
    def _generate_sample(self, idx, rng):
        # Loading a file isn't "generating"!
        return torch.load(f"sample_{idx}.pt")  # Wrong class!
```

**Why bad:** More complex than needed, misleading API

**Solution:** Use FileBasedInductiveDataset for loading files

---

## Naming Clarification

### Why is it called "Generated" if Papers100M isn't synthetic?

The name refers to **generating the sample object**, not generating the underlying data:

```python
# Papers100M (real data, not synthetic):
def _generate_sample(self, idx, rng):
    # "Generate" the Data object by:
    # 1. Extracting relevant portion from memory-mapped real data
    # 2. Creating a new PyG Data object
    # 3. Returning a fresh sample
    return Data(x=real_features[subset], ...)  # "Generated" Data object
```

**Alternative mental model:**
- **FileBasedInductiveDataset** = "LoadedInductiveDataset" (loads existing files)
- **OnDemandInductiveDataset** = "ComputedInductiveDataset" (computes on-demand)

The current name is slightly misleading but consistent with PyG's convention of "generated" meaning "not pre-loaded".

---

## Summary Decision Matrix

| Your Situation | Use This | Reason |
|---------------|----------|--------|
| Samples saved as files | `FileBasedInductiveDataset` | Just load them |
| Small dataset, need preprocessing | Preprocess → `FileBasedInductiveDataset` | Standard pipeline |
| Massive single graph | `OnDemandInductiveDataset` | Extract subgraphs on-demand |
| Synthetic/random data | `OnDemandInductiveDataset` | Cheaper to compute |
| Memory-mapped large data | `OnDemandInductiveDataset` | Load portions as needed |
| Need exact reproducibility | Either (both support it) | Use same seed |

---

## For Category B.1 Challenge

**Best practice:** Use BOTH to demonstrate complete understanding

1. **Preprocessing phase**: Use `OnDiskInductivePreprocessor`
   - Processes samples one-by-one (O(1) memory)
   - Saves to disk as individual files

2. **Loading phase**: Use `FileBasedInductiveDataset`
   - Loads preprocessed samples efficiently
   - Lightweight pickling for multi-worker DataLoader

3. **Massive graph demo**: Use `OnDemandInductiveDataset`
   - Shows scalability to 111M+ nodes (Papers100M)
   - Memory-mapping for O(1) memory

This demonstrates you understand:
- ✅ Memory-efficient preprocessing
- ✅ Memory-efficient loading
- ✅ Handling both "many small graphs" and "few giant graphs"
- ✅ Complete end-to-end pipeline

---

## Questions?

**Q: Should I always use these classes?**  
A: No! For small datasets (< 1GB), simple `InMemoryDataset` is fine. Use these for large-scale challenges.

**Q: Can I mix both approaches?**  
A: Yes! Use `OnDemandInductiveDataset` for raw data, preprocess, then use `FileBasedInductiveDataset` for training.

**Q: Why not just use PyG's OnDiskDataset?**  
A: PyG's OnDiskDataset uses SQLite backend which has:
- Higher overhead for pickling
- Slower parallel processing
- More complexity

Our approach is simpler and faster for the inductive setting.

**Q: The name "Generated" is confusing for Papers100M!**  
A: Agreed! Post-challenge, consider renaming to `ComputedInductiveDataset` or `OnDemandInductiveDataset`. For now, documentation clarifies the use cases.
