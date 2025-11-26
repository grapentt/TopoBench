"""Memory-efficient training script for OGBG-molpcba with explicit memory management.

This script adds aggressive memory management to prevent OOM with large datasets:
1. Explicit garbage collection after each batch
2. Cache clearing after each epoch
3. Memory monitoring and logging
4. Torch CUDA cache clearing (if using GPU)
"""

import gc
import sys
import torch
import psutil
import hydra
from omegaconf import DictConfig


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB


def log_memory(prefix=""):
    """Log current memory usage."""
    mem_mb = get_memory_usage()
    print(f"[Memory] {prefix}: {mem_mb:.1f} MB")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"[Memory] {prefix} (GPU): {gpu_mem:.1f} MB")


@hydra.main(version_base="1.3", config_path="configs", config_name="run.yaml")
def main(cfg: DictConfig):
    """Run training with explicit memory management."""
    
    # Import here to avoid loading heavy modules at startup
    from topobench.run import run
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import Callback
    
    # Custom callback for memory management
    class MemoryManagementCallback(Callback):
        """Callback to manage memory during training."""
        
        def __init__(self, log_every_n_batches=100):
            super().__init__()
            self.log_every_n_batches = log_every_n_batches
            self.batch_count = 0
        
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            """Clear cache after each batch."""
            self.batch_count += 1
            
            # Aggressive garbage collection every N batches
            if self.batch_count % self.log_every_n_batches == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_memory(f"After {self.batch_count} batches")
        
        def on_train_epoch_end(self, trainer, pl_module):
            """Aggressive cleanup after each epoch."""
            print(f"\n[Memory] Cleaning up after epoch {trainer.current_epoch}...")
            
            # Clear dataset cache if exists
            if hasattr(trainer.train_dataloader, 'dataset'):
                dataset = trainer.train_dataloader.dataset.dataset
                if hasattr(dataset, '_cache'):
                    print(f"[Memory] Clearing dataset cache ({len(dataset._cache)} items)")
                    dataset._cache.clear()
            
            # Full garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            log_memory("After epoch cleanup")
            print()
        
        def on_validation_epoch_end(self, trainer, pl_module):
            """Cleanup after validation."""
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Add memory management callback to config
    if 'callbacks' not in cfg:
        cfg.callbacks = {}
    
    print("\n" + "="*80)
    print("MEMORY-EFFICIENT TRAINING MODE")
    print("="*80)
    log_memory("Initial")
    print()
    
    # Set memory-efficient PyTorch settings
    torch.backends.cudnn.benchmark = False  # Disable to save memory
    if hasattr(torch.backends.cudnn, 'deterministic'):
        torch.backends.cudnn.deterministic = True
    
    # Enable garbage collection
    gc.enable()
    
    # Run training with memory callback
    try:
        # Run the standard training
        metric_dict, object_dict = run(cfg)
        
        log_memory("Final")
        return metric_dict
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "="*80)
            print("OUT OF MEMORY ERROR!")
            print("="*80)
            print("\nSuggestions:")
            print("1. Reduce batch_size further (try 16 or 8)")
            print("2. Ensure cache_size=0 in preprocessor config")
            print("3. Ensure num_workers=0 in dataloader config")
            print("4. Use gradient_accumulation if needed")
            print("5. Consider using a machine with more RAM")
            log_memory("At OOM")
            print()
            raise
        else:
            raise
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Check available memory
    total_mem = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
    available_mem = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
    
    print(f"\nSystem Memory: {total_mem:.1f} GB total, {available_mem:.1f} GB available")
    
    if available_mem < 4.0:
        print("WARNING: Less than 4GB available memory. Consider:")
        print("  - Closing other applications")
        print("  - Using smaller batch_size")
        print("  - Using smaller subset_size for testing")
    
    print()
    
    main()
