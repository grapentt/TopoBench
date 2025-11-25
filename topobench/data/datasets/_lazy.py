"""Lazy dataset splits for O(1) memory usage.

This module provides memory-efficient dataset splits that store only indices
rather than loading data into memory. This is critical for large on-disk datasets
where loading split metadata would consume significant memory.
"""

import torch
import torch_geometric
from torch.utils.data import Dataset


class LazySubset(Dataset):
    """Lazy subset that stores only indices, not data.

    Provides O(1) memory usage for dataset splits by storing indices instead of
    loading actual data into memory. This is critical for large on-disk datasets
    where loading split metadata would consume significant memory.

    The subset maps subset indices to source dataset indices, enabling efficient
    random access without memory overhead.

    Parameters
    ----------
    dataset : Dataset
        Source dataset to create subset from.
    indices : List[int] or torch.Tensor
        Indices for this subset.

    Examples
    --------
    >>> from topobench.data.datasets import LazySubset
    >>> dataset = OnDiskInductivePreprocessor(...)
    >>> train_indices = [0, 1, 2, 3, 4]
    >>> train_split = LazySubset(dataset, train_indices)
    >>> len(train_split)  # O(1) - just length of indices
    5
    >>> sample = train_split[0]  # O(1) - direct access via index
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int] | torch.Tensor,
    ):
        """Initialize lazy subset.

        Parameters
        ----------
        dataset : Dataset
            Source dataset.
        indices : List[int] or torch.Tensor
            Indices for this subset.
        """
        self.dataset = dataset

        # Convert to list for consistent behavior
        if isinstance(indices, torch.Tensor):
            self.indices = indices.tolist()
        else:
            self.indices = list(indices)

    def __len__(self) -> int:
        """Return length of subset.

        O(1) operation - just return length of index list.

        Returns
        -------
        int
            Number of samples in subset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Get sample at index.

        O(1) operation - direct lookup via stored index.

        Parameters
        ----------
        idx : int
            Index within subset (0 to len-1).

        Returns
        -------
        Data
            Sample from source dataset.

        Raises
        ------
        IndexError
            If idx is out of range for this subset.
        """
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(
                f"Index {idx} out of range for subset of size {len(self.indices)}"
            )

        # Map subset index to dataset index
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            String representation of subset.
        """
        return (
            f"LazySubset(dataset={self.dataset.__class__.__name__}, "
            f"size={len(self.indices)})"
        )


class LazyDataloadDataset(torch_geometric.data.Dataset):
    """Memory-efficient dataset split compatible with TBDataloader collate_fn.

    This class combines the O(1) memory efficiency of LazySubset with the tuple
    interface expected by DataloadDataset. It stores only indices and loads samples
    on-demand, then unpacks them into (values, keys) tuples for compatibility with
    the TopoBench dataloader pipeline.

    Parameters
    ----------
    dataset : Dataset
        Source dataset to create subset from (e.g., OnDiskInductivePreprocessor).
    indices : List[int] or torch.Tensor
        Indices for this subset.

    Examples
    --------
    >>> from topobench.data.datasets import LazyDataloadDataset
    >>> from topobench.data.preprocessor import OnDiskInductivePreprocessor
    >>>
    >>> # Create on-disk dataset
    >>> dataset = OnDiskInductivePreprocessor(...)
    >>>
    >>> # Create memory-efficient split
    >>> train_indices = [0, 1, 2, 3, 4]
    >>> train_split = LazyDataloadDataset(dataset, train_indices)
    >>>
    >>> # Use with TBDataloader - works seamlessly!
    >>> from topobench.dataloader import TBDataloader
    >>> datamodule = TBDataloader(train_split, val_split, test_split, batch_size=32)
    >>>
    >>> # Memory usage: O(1) - only stores indices
    >>> print(len(train_split))  # Fast: just returns len(indices)
    5
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int] | torch.Tensor,
    ):
        """Initialize lazy dataload dataset.

        Parameters
        ----------
        dataset : Dataset
            Source dataset.
        indices : List[int] or torch.Tensor
            Indices for this subset.
        """
        super().__init__()
        self.dataset = dataset

        # Convert to list for consistent behavior
        if isinstance(indices, torch.Tensor):
            self.indices = indices.tolist()
        else:
            self.indices = list(indices)

    def len(self) -> int:
        """Return length of subset.

        O(1) operation - just return length of index list.

        Returns
        -------
        int
            Number of samples in subset.
        """
        return len(self.indices)

    def get(self, idx: int) -> tuple[list, list]:
        """Get sample at index as (values, keys) tuple.

        This method unpacks Data objects into tuples compatible with collate_fn.
        The tuple format enables the collate function to properly batch samples.

        O(1) operation - direct lookup via stored index + tuple unpacking.

        Parameters
        ----------
        idx : int
            Index within subset (0 to len-1).

        Returns
        -------
        tuple
            (values, keys) where:
            - values: list of tensor values from the Data object
            - keys: list of corresponding attribute names

        Raises
        ------
        IndexError
            If idx is out of range for this subset.
        ```
        """
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(
                f"Index {idx} out of range for subset of size {len(self.indices)}"
            )

        # Map subset index to dataset index (LazySubset pattern)
        actual_idx = self.indices[idx]

        # Load sample from source dataset
        data = self.dataset[actual_idx]

        # Unpack into tuple format (DataloadDataset pattern)
        keys = list(data.keys())
        values = [data[key] for key in keys]

        return (values, keys)

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            String representation of subset.
        """
        return (
            f"LazyDataloadDataset(dataset={self.dataset.__class__.__name__}, "
            f"size={len(self.indices)})"
        )
