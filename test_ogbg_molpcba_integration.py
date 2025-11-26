"""Quick integration test for OGBG-molpcba dataset.

This script verifies that all components work together correctly:
- Dataset loading (mock mode)
- On-disk preprocessing
- Model training (1 epoch)

Run this before attempting full training to catch any issues early.
"""

from pathlib import Path
from omegaconf import OmegaConf
import lightning as pl

from topobench.data.loaders import OGBGMolPCBALoader
from topobench.data.preprocessor import OnDiskInductivePreprocessor
from topobench.dataloader import TBDataloader
from topobench.model import TBModel
from topomodelx.nn.simplicial.scn2 import SCN2
from topobench.nn.wrappers.simplicial import SCNWrapper
from topobench.nn.readouts import PropagateSignalDown
from topobench.nn.encoders import AllCellFeatureEncoder
from topobench.loss import TBLoss
from topobench.optimizer import TBOptimizer
from topobench.evaluator.evaluator import TBEvaluator


def test_ogbg_molpcba_integration():
    """Test complete pipeline with mock data."""
    print("=" * 80)
    print("OGBG-molpcba Integration Test")
    print("=" * 80)
    
    # Use temporary directory for test
    test_dir = Path("./test_data_ogbg_molpcba")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Load mock dataset
        print("\n[1/6] Loading mock dataset...")
        loader_config = OmegaConf.create({
            "data_dir": str(test_dir),
            "data_name": "ogbg-molpcba",
            "split": "train",
            "subset_size": 50,  # Small for fast testing
            "use_mock": True,  # No download needed
        })
        
        loader = OGBGMolPCBALoader(loader_config)
        source_dataset, _ = loader.load()
        
        assert len(source_dataset) == 50, f"Expected 50 samples, got {len(source_dataset)}"
        sample = source_dataset[0]
        assert hasattr(sample, 'x'), "Sample missing node features"
        assert hasattr(sample, 'edge_index'), "Sample missing edge index"
        assert hasattr(sample, 'y'), "Sample missing labels"
        assert sample.y.shape[-1] == 128, f"Expected 128 labels, got {sample.y.shape[-1]}"
        print(f"✓ Loaded {len(source_dataset)} mock molecules")
        print(f"  Sample: {sample.num_nodes} nodes, {sample.edge_index.shape[1]} edges")
        
        # Step 2: Configure transforms
        print("\n[2/6] Configuring transforms...")
        transforms_config = OmegaConf.create({
            "clique_lifting": {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": 2,
            }
        })
        print("✓ Transform: SimplicialCliqueLifting (dim=2)")
        
        # Step 3: On-disk preprocessing
        print("\n[3/6] Applying on-disk preprocessing...")
        preprocessed = OnDiskInductivePreprocessor(
            dataset=source_dataset,
            data_dir=test_dir / "preprocessed",
            transforms_config=transforms_config,
            storage_backend="mmap",  # Use mmap backend (default)
            num_workers=None,
            force_reload=True,  # Always reprocess for test
            cache_size=0,  # Disable cache for testing
        )
        
        assert len(preprocessed) == 50, f"Expected 50 preprocessed samples, got {len(preprocessed)}"
        preprocessed_sample = preprocessed[0]
        assert hasattr(preprocessed_sample, 'x_0'), "Missing 0-cells"
        assert hasattr(preprocessed_sample, 'x_1'), "Missing 1-cells"
        assert hasattr(preprocessed_sample, 'x_2'), "Missing 2-cells"
        print(f"✓ Preprocessing complete: {len(preprocessed)} samples")
        print(f"  0-cells: {preprocessed_sample.x_0.shape}")
        print(f"  1-cells: {preprocessed_sample.x_1.shape}")
        print(f"  2-cells: {preprocessed_sample.x_2.shape}")
        
        # Step 4: Create splits
        print("\n[4/6] Creating dataset splits...")
        split_config = OmegaConf.create({
            "learning_setting": "inductive",
            "split_type": "random",
            "data_seed": 42,
            "data_split_dir": str(test_dir / "splits"),
            "train_prop": 0.8,
            "val_prop": 0.1,
        })
        
        train_ds, val_ds, test_ds = preprocessed.load_dataset_splits(split_config)
        print(f"✓ Splits created:")
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Val:   {len(val_ds)} samples")
        print(f"  Test:  {len(test_ds)} samples")
        
        # Step 5: Build model
        print("\n[5/6] Building SCN2 model...")
        HIDDEN_DIM = 32  # Small for fast testing
        NUM_CLASSES = 128
        NUM_FEATURES = 9
        
        feature_encoder = AllCellFeatureEncoder(
            in_channels=[NUM_FEATURES, NUM_FEATURES, NUM_FEATURES],
            out_channels=HIDDEN_DIM,
        )
        
        backbone = SCN2(
            in_channels_0=HIDDEN_DIM,
            in_channels_1=HIDDEN_DIM,
            in_channels_2=HIDDEN_DIM,
        )
        
        def wrapper_factory(**kwargs):
            def factory(backbone):
                return SCNWrapper(backbone, **kwargs)
            return factory
        
        backbone_wrapper = wrapper_factory(
            out_channels=HIDDEN_DIM,
            num_cell_dimensions=3,
        )
        
        readout = PropagateSignalDown(
            readout_name="mean",
            num_cell_dimensions=3,
            hidden_dim=HIDDEN_DIM,
            out_channels=NUM_CLASSES,
            task_level="graph",
        )
        
        model = TBModel(
            backbone=backbone,
            backbone_wrapper=backbone_wrapper,
            readout=readout,
            loss=TBLoss(dataset_loss={"task": "multilabel classification", "loss_type": "BCE"}),
            feature_encoder=feature_encoder,
            evaluator=TBEvaluator(
                task="multilabel classification",
                num_classes=NUM_CLASSES,
                metrics=["accuracy", "f1_macro"]
            ),
            optimizer=TBOptimizer(optimizer_id="Adam", parameters={"lr": 0.001}),
            compile=False,
        )
        print("✓ Model created (SCN2, hidden_dim=32)")
        
        # Step 6: Train for 1 epoch
        print("\n[6/6] Training for 1 epoch...")
        datamodule = TBDataloader(
            train_ds, val_ds, test_ds,
            batch_size=8,
            num_workers=0,
        )
        
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )
        
        trainer.fit(model, datamodule)
        print("✓ Training complete")
        
        # Test
        test_results = trainer.test(model, datamodule, verbose=False)
        print(f"✓ Test complete: {test_results[0]}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nIntegration verified:")
        print("  ✓ Dataset loading works")
        print("  ✓ On-disk preprocessing works")
        print("  ✓ Model training works")
        print("  ✓ Evaluation works")
        print("\nYou're ready to train on the full dataset!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\n✓ Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    success = test_ogbg_molpcba_integration()
    exit(0 if success else 1)
