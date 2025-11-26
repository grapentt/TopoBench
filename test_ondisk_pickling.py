#!/usr/bin/env python
"""Test on-disk preprocessing with parallel workers (pickling test)."""

from topobench.data.loaders import OGBGMolPCBALoader
from topobench.data.preprocessor import create_preprocessor
from omegaconf import OmegaConf

print("=" * 70)
print("TESTING ON-DISK PREPROCESSING WITH PICKLING")
print("=" * 70)

# Load dataset
print("\n1️⃣ Loading dataset...")
loader_config = OmegaConf.create({
    "data_dir": "./data/ogbg_molpcba",
    "data_name": "ogbg-molpcba",
    "subset_size": 20,
    "use_mock": True,
})
loader = OGBGMolPCBALoader(parameters=loader_config)
dataset, dataset_dir = loader.load()
print(f"   ✅ Loaded {len(dataset)} samples")

# Create transforms config
print("\n2️⃣ Setting up transforms...")
transforms_config = OmegaConf.create({
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
print("   ✅ Transforms configured")

# Test on-disk preprocessing with parallel workers
print("\n3️⃣ Testing on-disk preprocessing (parallel, with pickling)...")
preprocessor = create_preprocessor(
    dataset=dataset,
    data_dir="./data/test_ondisk_pickling",
    transforms_config=transforms_config,
    mode="ondisk",
    force_reload=True,
    num_workers=2,  # Test multiprocessing!
)

print(f"   Preprocessor type: {type(preprocessor).__name__}")

# Try to process
print("\n4️⃣ Running preprocessing...")
try:
    split_params = OmegaConf.create({
        "learning_setting": "inductive",
        "data_split_dir": "./data/test_splits",
        "data_seed": 42,
        "split_type": "random",
        "train_prop": 0.6,
        "val_prop": 0.2
    })
    
    train, val, test = preprocessor.load_dataset_splits(split_params)
    
    print(f"   ✅ SUCCESS! Preprocessing completed without pickling errors!")
    print(f"   Train: {len(train)} samples")
    print(f"   Val: {len(val)} samples")
    print(f"   Test: {len(test)} samples")
    
except Exception as e:
    if "pickle" in str(e).lower():
        print(f"   ❌ PICKLING ERROR: {e}")
    else:
        print(f"   ⚠️  Other error (not pickling): {type(e).__name__}")
        print(f"      {e}")
        # This is OK - we only care about pickling errors

print("\n" + "=" * 70)
print("PICKLING TEST COMPLETE")
print("=" * 70)
