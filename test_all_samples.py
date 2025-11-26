#!/usr/bin/env python
"""Test all 100 mock samples to find which one fails."""

from topobench.data.datasets.ogbg_molpcba import MockMolecularDataset
from topobench.transforms.liftings.graph2simplicial import SimplicialCliqueLifting

print("=" * 70)
print("TESTING ALL 100 MOCK SAMPLES")
print("=" * 70)

# Create mock dataset with 100 samples (same as config)
dataset = MockMolecularDataset(root="./data/test_all_mock", num_samples=100, seed=42)
lifting = SimplicialCliqueLifting(complex_dim=2)

print(f"\nTesting {len(dataset)} samples...")

failed_samples = []
for idx in range(len(dataset)):
    try:
        data = dataset[idx]
        lifted_data = lifting(data)
        if (idx + 1) % 10 == 0:
            print(f"   ✅ Samples 0-{idx}: OK")
    except Exception as e:
        failed_samples.append((idx, str(e)))
        print(f"   ❌ Sample {idx} FAILED: {e}")

if not failed_samples:
    print(f"\n✅ ALL {len(dataset)} SAMPLES PASSED!")
else:
    print(f"\n❌ {len(failed_samples)} samples failed:")
    for idx, error in failed_samples[:10]:
        print(f"   Sample {idx}: {error}")

print("=" * 70)
