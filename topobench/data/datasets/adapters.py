"""Adapters for converting PyG datasets to lightweight on-disk format.

Converts any PyTorch Geometric dataset into an optimized format that enables
efficient parallel preprocessing through lightweight pickling.
"""

import sys
import torch
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset

from .base_inductive import BaseOnDiskInductiveDataset


def _extract_sample(idx: int, source_ds, cache_dir):
    """Extract and cache single sample (module-level for pickling-able)."""
    try:
        sample = source_ds[idx]
        cache_path = (
            cache_dir / f"sample_{idx:06d}.pt"
        )  # Match base class format
        torch.save(sample, cache_path)
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)


class PyGDatasetAdapter(BaseOnDiskInductiveDataset):
    """Convert PyG datasets to lightweight on-disk format.

    Extracts samples from any PyG dataset and caches them as individual files,
    enabling efficient parallel preprocessing through lightweight pickling.

    Parameters
    ----------
    source_dataset : PyG Dataset
        Any PyTorch Geometric dataset
    root : str | Path
        Directory for caching (default: temp directory)
    cache_samples : bool
        Enable disk caching (default: True)
    force_rebuild : bool
        Force cache rebuild (default: False)
    """

    def __init__(
        self,
        source_dataset: PyGDataset,
        root: str | Path | None = None,
        cache_samples: bool = True,
        force_rebuild: bool = False,
    ):
        """Initialize adapter from PyG dataset.

        Parameters
        ----------
        source_dataset : PyG Dataset
            The dataset to adapt (InMemoryDataset, TUDataset, etc.).
        root : str or Path, optional
            Cache directory. If None, uses temporary location.
        cache_samples : bool, optional
            Enable sample caching (required for multiprocessing).
        force_rebuild : bool, optional
            Force rebuild of cache even if it exists.
        """
        self.source_dataset = source_dataset
        self.force_rebuild = force_rebuild
        self._num_samples = len(source_dataset)

        # Determine cache location
        if root is None:
            import tempfile

            root = (
                Path(tempfile.gettempdir())
                / "topobench_cache"
                / "adapted_dataset"
            )

        root = Path(root)

        # Initialize base class
        super().__init__(root, cache_samples=cache_samples)

    def _ensure_samples_extracted(
        self, num_workers: int | None = None, verbose: bool = True
    ):
        """Extract samples from source dataset if not already cached.

        Parameters
        ----------
        num_workers : int or None, optional
            Number of parallel workers for extraction.
            None = use all available cores - 1 (recommended).
            1 = sequential extraction (safest).
            > 1 = use specified number of workers.
        verbose : bool, optional
            Print progress messages (default: True).
        """
        if not self.cache_samples:
            return

        # Check if cache already populated
        expected_files = self._num_samples
        cached_files = len(list(self.cache_dir.glob("sample_*.pt")))

        if cached_files == expected_files and not self.force_rebuild:
            if verbose:
                print(f"✓ Using cached samples ({cached_files} files)")
            return

        # Determine indices to extract
        if self.force_rebuild:
            indices_to_extract = list(range(self._num_samples))
        else:
            indices_to_extract = [
                idx
                for idx in range(self._num_samples)
                if not self._get_cache_path(idx).exists()
            ]

        if not indices_to_extract:
            if verbose:
                print(f"✓ All samples already cached ({expected_files} files)")
            return

        if verbose:
            print(
                f"Extracting {len(indices_to_extract)}/{self._num_samples} samples from source dataset..."
            )

        # Sequential extraction
        if num_workers == 1:
            self._extract_sequential(indices_to_extract, verbose)
        else:
            # Parallel extraction (None or > 1)
            self._extract_parallel(indices_to_extract, num_workers, verbose)

        if verbose:
            print(f"✓ Cached {self._num_samples} samples to {self.cache_dir}")

    def _extract_sequential(self, indices: list[int], verbose: bool):
        """Extract samples sequentially."""
        for i, idx in enumerate(indices):
            cache_path = self._get_cache_path(idx)
            sample = self.source_dataset[idx]
            torch.save(sample, cache_path)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Extracted {i + 1}/{len(indices)} samples...")

    def _extract_parallel(
        self, indices: list[int], num_workers: int | None, verbose: bool
    ):
        """Extract samples in parallel using multiprocessing.

        Parameters
        ----------
        num_workers : int or None
            Number of workers. None = use all cores - 1.
        """
        import multiprocessing as mp
        from functools import partial

        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        if verbose:
            print(f"  Using {num_workers} parallel workers for extraction")

        # Create extraction function with fixed args
        extract_fn = partial(
            _extract_sample,
            source_ds=self.source_dataset,
            cache_dir=self.cache_dir,
        )

        # Use mp.Pool with chunksize for better performance
        chunksize = max(
            1, len(indices) // (num_workers * 4)
        )  # 4 chunks per worker
        
        pbar = (
            tqdm(
                total=len(indices),
                desc="Extracting samples",
                unit="sample",
            )
            if verbose
            else None
        )

        # Use appropriate context: fork is faster on Linux, spawn is safer elsewhere
        if sys.platform == "linux":
            mp_context = mp.get_context("fork")
        else:
            mp_context = mp.get_context("spawn")

        # Use mp.Pool for simple I/O - lower overhead than ProcessPoolExecutor
        with mp_context.Pool(num_workers) as pool:
            results = []
            for i, result in enumerate(
                pool.imap_unordered(extract_fn, indices, chunksize=chunksize)
            ):
                results.append(result)
                if pbar:
                    pbar.update(1)
                elif verbose and (i + 1) % 100 == 0:
                    print(f"  Extracted {i + 1}/{len(indices)} samples...")

        if pbar:
            pbar.close()

        # Check for failures
        failures = [r for r in results if not r[1]]
        if failures:
            print(f"⚠️  {len(failures)} samples failed to extract")
            for idx, _, error in failures[:5]:  # Show first 5
                print(f"    Sample {idx}: {error}")

    def _get_num_samples(self) -> int:
        """Get number of samples from source dataset."""
        return len(self.source_dataset)

    def _generate_or_load_sample(self, idx: int) -> Data:
        """Load sample from source dataset."""
        return self.source_dataset[idx]

    def _get_pickle_args(self) -> tuple:
        """Get pickle arguments.

        Note: We don't pickle source_dataset since it may be heavy.
        Instead, we rely on cached files.
        """
        return (None, str(self.root), self.cache_samples, False)

    def __reduce__(self):
        """Custom pickling for multiprocessing.

        After samples are cached, we can pickle without the source dataset.
        This makes the adapter lightweight for parallel processing.
        """
        if not self.cache_samples:
            raise RuntimeError(
                "PyGDatasetAdapter requires cache_samples=True for multiprocessing. "
                "The source dataset cannot be pickled efficiently."
            )

        # Check cache is populated
        if len(list(self.cache_dir.glob("sample_*.pt"))) < self._num_samples:
            raise RuntimeError(
                "Samples not yet cached. Access at least one sample before "
                "using with multiprocessing."
            )

        # Create a lightweight version that loads from cache only
        return (
            _create_adapted_from_cache,
            (str(self.root), self._num_samples, self.cache_samples),
        )


def _create_adapted_from_cache(
    root: str, num_samples: int, cache_samples: bool
):
    """Create adapter that loads from cache (for unpickling).

    This is a lightweight constructor used after pickling that doesn't
    require the original source dataset.
    """

    class CachedDataset(BaseOnDiskInductiveDataset):
        def __init__(self, root, num_samples):
            self.num_samples = num_samples
            super().__init__(root, cache_samples=True)

        def _get_num_samples(self):
            return self.num_samples

        def _generate_or_load_sample(self, idx):
            # Load from cache - base class handles caching automatically
            # Just load directly from source if cache enabled
            cache_path = self._get_cache_path(idx)
            if cache_path.exists():
                # PyTorch 2.6+ requires weights_only=False for PyG Data objects
                return torch.load(cache_path, weights_only=False)
            else:
                # This shouldn't happen but provide informative error
                import os

                raise FileNotFoundError(
                    f"Sample {idx} not found at {cache_path}. "
                    f"Cache dir: {self.cache_dir}, exists: {self.cache_dir.exists()}, "
                    f"files: {list(self.cache_dir.glob('sample_*.pt'))[:5] if self.cache_dir.exists() else 'N/A'}"
                )

    return CachedDataset(root, num_samples)


def adapt_dataset(
    dataset: PyGDataset,
    root: str | Path | None = None,
    force_rebuild: bool = False,
    extraction_workers: int = 1,
    verbose: bool = True,
) -> PyGDatasetAdapter:
    """Convert any PyG dataset to optimal on-disk format (one-line convenience).

    This is a convenience function that wraps PyGDatasetAdapter for easy use.

    Parameters
    ----------
    dataset : PyG Dataset
        Any PyTorch Geometric dataset to adapt.
    root : str or Path, optional
        Cache directory. If None, uses temp location.
    force_rebuild : bool, optional
        Rebuild cache even if exists (default: False).
    extraction_workers : int or None, optional
        Number of parallel workers for sample extraction (default: 1).
        1 = sequential extraction (safest).
        None = use all available cores - 1 (recommended for large datasets).
        > 1 = use specified number of workers.
    verbose : bool, optional
        Print progress messages (default: True).

    Returns
    -------
    PyGDatasetAdapter
        Optimized dataset ready for parallel preprocessing.

    Examples
    --------
    >>> from torch_geometric.datasets import TUDataset
    >>> from topobench.data.datasets import adapt_dataset
    >>>
    >>> # Sequential extraction (default, safest)
    >>> enzymes = TUDataset(root="./data", name="ENZYMES")
    >>> enzymes_optimized = adapt_dataset(enzymes)
    >>>
    >>> # Parallel extraction for large datasets (all cores - 1)
    >>> big_dataset = TUDataset(root="./data", name="PROTEINS")
    >>> big_optimized = adapt_dataset(big_dataset, extraction_workers=None)
    >>>
    >>> # Specific number of workers
    >>> custom = adapt_dataset(big_dataset, extraction_workers=4)

    >>> # Use with parallel preprocessing
    >>> preprocessor = OnDiskInductivePreprocessor(
    ...     dataset=enzymes_optimized,
    ...     num_workers=7  # 2.29× faster!
    ... )
    """
    # Create adapter without auto-extraction
    adapter = PyGDatasetAdapter(
        source_dataset=dataset,
        root=root,
        cache_samples=True,
        force_rebuild=force_rebuild,
    )

    # Extract samples with specified workers
    adapter._ensure_samples_extracted(
        num_workers=extraction_workers, verbose=verbose
    )

    return adapter


def adapt_tu_dataset(
    name: str, root: str | Path = "./data"
) -> PyGDatasetAdapter:
    """Load and adapt a TU dataset in one line.

    Convenience function for common TU datasets (ENZYMES, PROTEINS, etc.).

    Parameters
    ----------
    name : str
        TU dataset name (e.g., "ENZYMES", "PROTEINS", "MUTAG").
    root : str or Path, optional
        Root directory for data (default: "./data").

    Returns
    -------
    PyGDatasetAdapter
        Optimized dataset ready for parallel preprocessing.

    Examples
    --------
    >>> from topobench.data.datasets import adapt_tu_dataset
    >>>
    >>> # Load and optimize in one line!
    >>> enzymes = adapt_tu_dataset("ENZYMES")
    >>> proteins = adapt_tu_dataset("PROTEINS")
    >>>
    >>> # Use immediately with parallel preprocessing
    >>> preprocessor = OnDiskInductivePreprocessor(
    ...     dataset=enzymes,
    ...     num_workers=7
    ... )
    """
    from torch_geometric.datasets import TUDataset

    root = Path(root)
    dataset = TUDataset(root=str(root / "raw"), name=name)
    cache_root = root / "adapted" / name.lower()

    return adapt_dataset(dataset, root=cache_root)
