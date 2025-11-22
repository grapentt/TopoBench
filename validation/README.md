# Validation Scripts

This directory contains production-ready validation scripts that demonstrate the effectiveness of TopoBench's on-disk learning approaches.

## Purpose

These scripts provide **reproducible proof** that:
1. **In-memory preprocessing FAILS** (OOM) on large datasets
2. **On-disk preprocessing SUCCEEDS** with constant memory

Each script automatically calculates dataset parameters guaranteed to cause OOM based on available RAM, then demonstrates the same dataset succeeds with on-disk processing.

## Scripts

### Inductive Validation

**Script**: `validate_inductive_ondisk.py`

Demonstrates on-disk inductive learning (multiple graphs).

```bash
# Basic usage (assumes 4GB RAM available)
python validation/validate_inductive_ondisk.py

# Custom RAM limit
python validation/validate_inductive_ondisk.py --max_ram_gb 8.0

# Custom data directory
python validation/validate_inductive_ondisk.py \
    --max_ram_gb 4.0 \
    --data_dir ./validation_data
```

**What it does**:
1. Calculates dataset size that will exceed specified RAM
2. Attempts in-memory preprocessing → **OOM**
3. Runs on-disk preprocessing on same dataset → **SUCCESS**
4. Tracks memory usage throughout

### Transductive Validation

**Script**: `validate_transductive_ondisk.py`

Demonstrates on-disk transductive learning (single large graph).

```bash
# Basic usage
python validation/validate_transductive_ondisk.py

# Custom configuration
python validation/validate_transductive_ondisk.py \
    --max_ram_gb 4.0 \
    --data_dir ./validation_data
```

**What it does**:
1. Calculates graph size that will exceed specified RAM  
2. Attempts in-memory preprocessing → **OOM**
3. Builds structure index on-disk → **SUCCESS**
4. Verifies constant memory usage

## Features

### Automatic Dataset Sizing

Scripts use `calculate_oom_params()` to automatically determine dataset parameters:

```python
from topobench.utils.validation_utils import calculate_oom_params

# Calculate params for 4GB RAM system
params = calculate_oom_params(
    max_ram_gb=4.0,
    approach="inductive",  # or "transductive"
    structure_type="triangles"
)

# params contains: num_graphs, nodes_per_graph, estimated_memory_gb, etc.
```

### Memory Tracking

Built-in memory tracking with `MemoryTracker`:

```python
from topobench.utils.validation_utils import MemoryTracker

tracker = MemoryTracker()
with tracker:
    # Your code here
    process_data()

tracker.report()  # Prints memory usage report
```

### Expected Behavior Testing

Use `expect_oom()` to verify functions fail as expected:

```python
from topobench.utils.validation_utils import expect_oom

def will_oom():
    huge_tensor = torch.randn(100000, 100000)

oom_occurred, msg = expect_oom(will_oom)
assert oom_occurred, "Expected OOM!"
```

## Output Example

```
================================================================================
                     Inductive On-Disk Validation
================================================================================

Configuration:
  - Max RAM for OOM test: 4.0GB
  - Data directory: ./data/validation_inductive

Calculating dataset parameters...
✓ Parameters calculated to exceed 4.0GB RAM

--------------------------------------------------------------------------------
[1/2] Testing In-Memory Approach (Expected: OOM)
--------------------------------------------------------------------------------

Dataset parameters:
  - Graphs: 500
  - Nodes per graph: 100
  - Estimated structures: 2,000,000
  - Estimated memory: 6.00GB

  ✓ In-memory approach failed: OutOfMemoryError

--------------------------------------------------------------------------------
[2/2] Testing On-Disk Approach (Expected: SUCCESS)
--------------------------------------------------------------------------------

Loading dataset...
✓ Loaded 500 graphs

Processing with on-disk approach...
✓ Processed 500 samples

Verifying sample loading...
  Sample 0: 100 nodes
  Sample 1: 100 nodes
  Sample 2: 100 nodes
✓ Successfully loaded samples

On-Disk Memory Usage Report:
  RAM: 500MB → 650MB (Δ 150MB)
  GPU: 0MB → 0MB (Δ 0MB)

  ✓ Memory stayed constant: 0.15GB increase (vs 6.00GB for in-memory)

================================================================================
                          Validation Results
================================================================================

Test 1: In-Memory Approach
  ✓ Failed with OOM as expected

Test 2: On-Disk Approach
  ✓ Succeeded with constant memory

Conclusion:
  ✓ ✓ VALIDATION PASSED: On-disk enables training on datasets that don't fit in memory!
```

## Requirements

- `psutil` for memory tracking
- `torch` and `torch_geometric`
- TopoBench with on-disk support

Install:
```bash
pip install psutil torch torch-geometric
```

## Utilities

All validation utilities are in `topobench/utils/validation_utils.py`:

- **`MemoryTracker`**: Track RAM/GPU memory usage
- **`calculate_oom_params()`**: Auto-calculate dataset params for OOM
- **`expect_oom()`**: Test that functions OOM as expected
- **`print_section()`, `print_subsection()`, `print_result()`**: Formatted output

## Integration with TopoBench

These scripts follow TopoBench patterns:
- Use `AbstractLoader` pattern for dataset loading
- Support `OmegaConf` configuration
- Compatible with TopoBench preprocessors
- Follow PEP8 and type hints

## Customization

To validate with different configurations:

```python
# Custom structure types
params = calculate_oom_params(
    max_ram_gb=8.0,
    approach="inductive",
    structure_type="4-cliques"  # Instead of triangles
)

# Custom lifting config
transforms_config = OmegaConf.create({
    "custom_lifting": {
        "transform_type": "lifting",
        "transform_name": "YourCustomLifting",
        "param1": value1,
    }
})
```

## Troubleshooting

**Problem**: Script doesn't OOM as expected

**Solutions**:
- Increase `max_ram_gb` (script calculates 1.5x overage, might not be enough)
- Manually increase dataset size in script
- Check available RAM: `free -g` (Linux) or Activity Monitor (Mac)

**Problem**: On-disk also fails

**Causes**:
- Disk space full (check with `df -h`)
- Permissions issue (check write access to data_dir)
- Bug in on-disk implementation (check logs)

**Problem**: Takes too long

**Solutions**:
- Reduce dataset size (edit `calculate_oom_params` output)
- Use faster storage (SSD instead of HDD)
- Skip validation for smaller RAM systems

## Citation

If you use these validation scripts, please cite:

```bibtex
@software{topobench_ondisk_2024,
  title={TopoBench On-Disk: Memory-Efficient Topological Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/your/repo}
}
```

## License

Same as TopoBench main project.
