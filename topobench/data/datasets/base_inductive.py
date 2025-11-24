"""Lightweight inductive dataset classes for efficient parallel preprocessing.

Provides base classes that enable parallel speedup through lightweight pickling
and on-demand loading. All classes maintain O(1) memory usage and support caching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class BaseOnDiskInductiveDataset(Dataset, ABC):
    """Base class for lightweight on-disk datasets.
    
    Provides lightweight pickling, on-demand loading, and automatic caching.
    Subclasses must implement `_get_num_samples()` and `_generate_or_load_sample(idx)`.
    
    Parameters
    ----------
    root : str | Path
        Root directory for caching
    cache_samples : bool
        Enable disk caching (default: True)
    """
    
    def __init__(self, root: str | Path, cache_samples: bool = True):
        self.root = Path(root)
        self.cache_samples = cache_samples
        
        if self.cache_samples:
            self.cache_dir = self.root / ".sample_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    @abstractmethod
    def _get_num_samples(self) -> int:
        """Return total number of samples."""
        pass
    
    @abstractmethod
    def _generate_or_load_sample(self, idx: int) -> Data:
        """Generate or load a single sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        Data
            The sample data
        """
        pass
    
    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for sample index."""
        return self.cache_dir / f"sample_{idx:06d}.pt"
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self._get_num_samples()
    
    def __getitem__(self, idx: int | slice) -> Data | list[Data]:
        """Get sample(s) by index or slice.
        
        Parameters
        ----------
        idx : int or slice
            Sample index or slice
            
        Returns
        -------
        Data or list[Data]
            Single sample or list of samples
        """
        # Handle slicing
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step or 1)]
        
        # Handle negative indexing
        if idx < 0:
            idx = len(self) + idx
        
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        
        # Try loading from cache
        if self.cache_samples:
            cache_path = self._get_cache_path(idx)
            if cache_path.exists():
                return torch.load(cache_path, weights_only=False)
        
        # Generate/load sample
        sample = self._generate_or_load_sample(idx)
        
        # Save to cache
        if self.cache_samples:
            cache_path = self._get_cache_path(idx)
            torch.save(sample, cache_path)
        
        return sample
    
    def _get_pickle_args(self) -> tuple:
        """Get arguments for pickling.
        
        Override this in subclasses to include additional attributes.
        Base implementation returns (root, cache_samples).
        """
        return (str(self.root), self.cache_samples)
    
    def __reduce__(self):
        """Support pickling for multiprocessing."""
        return (self.__class__, self._get_pickle_args())


class FileBasedInductiveDataset(BaseOnDiskInductiveDataset):
    """Dataset that loads samples from individual files.
    
    Automatically discovers files matching the pattern and provides
    lightweight access for parallel processing.
    
    Parameters
    ----------
    root : str | Path
        Root directory containing sample files
    file_pattern : str
        Glob pattern for sample files (default: "*.pt")
    cache_samples : bool
        Enable disk caching (default: True)
    """
    
    def __init__(
        self,
        root: str | Path,
        file_pattern: str = "*.pt",
        cache_samples: bool = True
    ):
        self.file_pattern = file_pattern
        root_path = Path(root)
        self.files = sorted(list(root_path.glob(file_pattern)))
        super().__init__(root, cache_samples=cache_samples)
    
    def _get_num_samples(self) -> int:
        """Return number of discovered files."""
        return len(self.files)
    
    @abstractmethod
    def _load_file(self, file_path: Path) -> Data:
        """Load a single file.
        
        Subclasses implement their loading logic here.
        
        Parameters
        ----------
        file_path : Path
            Path to the file to load
            
        Returns
        -------
        Data
            The loaded sample
        """
        pass
    
    def _generate_or_load_sample(self, idx: int) -> Data:
        """Load sample from file."""
        return self._load_file(self.files[idx])
    
    def _get_pickle_args(self) -> tuple:
        """Include file_pattern in pickle args."""
        return (str(self.root), self.file_pattern, self.cache_samples)


class OnDemandInductiveDataset(BaseOnDiskInductiveDataset):
    """Dataset that generates samples on-demand.
    
    Generates samples deterministically using seeded random number generation.
    Suitable for synthetic data, procedural generation, or on-the-fly subgraph extraction.
    
    Parameters
    ----------
    root : str | Path
        Root directory for caching
    num_samples : int
        Total number of samples to generate
    seed : int
        Random seed for deterministic generation (default: 42)
    cache_samples : bool
        Enable disk caching (default: True)
    """
    
    def __init__(
        self,
        root: str | Path,
        num_samples: int,
        seed: int = 42,
        cache_samples: bool = True
    ):
        self.num_samples = num_samples
        self.seed = seed
        super().__init__(root, cache_samples=cache_samples)
    
    def _get_num_samples(self) -> int:
        """Return configured number of samples."""
        return self.num_samples
    
    @abstractmethod
    def _generate_sample(self, idx: int, rng: torch.Generator) -> Data:
        """Generate a single sample.
        
        Subclasses implement their generation logic here.
        
        Parameters
        ----------
        idx : int
            Sample index
        rng : torch.Generator
            Seeded random number generator for deterministic generation
            
        Returns
        -------
        Data
            The generated sample
        """
        pass
    
    def _generate_or_load_sample(self, idx: int) -> Data:
        """Generate sample with deterministic seeding."""
        # Create generator with deterministic seed
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)
        return self._generate_sample(idx, rng)
    
    def _get_pickle_args(self) -> tuple:
        """Include num_samples and seed in pickle args."""
        return (str(self.root), self.num_samples, self.seed, self.cache_samples)
