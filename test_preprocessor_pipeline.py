#!/usr/bin/env python
"""Test preprocessor pipeline."""

from omegaconf import OmegaConf
import hydra
from topobench.data.loaders.ogbg_molpcba_loader import OGBGMolPCBALoader
from topobench.data.preprocessor import PreProcessor
from topobench.transforms.liftings.graph2simplicial import SimplicialCliqueLifting
from topobench.transforms.feature_liftings import ProjectionSum
from torch_geometric.transforms import Compose

print("=" * 70)
print("TESTING PREPROCESSOR PIPELINE")
print("=" * 70)

# Load dataset
config = OmegaConf.create({
    "data_dir": "./data/test_preprocessor",
    "subset_size": 2,
    "split": "train",
    "use_mock": True,
})

loader = OGBGMolPCBALoader(config)
dataset, data_dir = loader.load()

print(f"\n1️⃣ Dataset loaded: {len(dataset)} samples")
data = dataset[0]
print(f"   Sample 0 edge_index shape: {data.edge_index.shape}")

# Create transforms
print(f"\n2️⃣ Creating transforms...")
lifting = SimplicialCliqueLifting(complex_dim=2)
projection = ProjectionSum()

# Test with Compose (what preprocessor uses)
print(f"\n3️⃣ Testing with Compose (preprocessor style)...")
try:
    composed = Compose([lifting, projection])
    transformed = composed(data)
    print(f"   ✅ SUCCESS with Compose!")
except Exception as e:
    print(f"   ❌ FAILED with Compose: {e}")
    import traceback
    traceback.print_exc()

# Test individual
print(f"\n4️⃣ Testing transforms individually...")
try:
    lifted = lifting(data)
    print(f"   ✅ Lifting SUCCESS!")
    projected = projection(lifted)
    print(f"   ✅ Projection SUCCESS!")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test with preprocessor config format
print(f"\n5️⃣ Testing with DataTransform wrapper...")
from topobench.transforms.data_transform import DataTransform

try:
    dt_lifting = DataTransform(transform_name="SimplicialCliqueLifting", complex_dim=2)
    dt_projection = DataTransform(transform_name="ProjectionSum")
    
    lifted2 = dt_lifting(data)
    print(f"   ✅ DataTransform lifting SUCCESS!")
    projected2 = dt_projection(lifted2)
    print(f"   ✅ DataTransform projection SUCCESS!")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
