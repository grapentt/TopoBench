"""On-disk preprocessor for inductive learning with large datasets.

This module provides a memory-efficient preprocessing implementation that processes
samples sequentially and stores them on disk, enabling training on datasets
larger than available RAM. This preprocessor applies transforms (e.g., lifting
operations from graphs to simplicial complexes) one sample at a time to maintain
constant memory usage.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch_geometric
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from topobench.data.utils import ensure_serializable, make_hash
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class OnDiskInductivePreprocessor(Dataset):
    """Sequential disk-backed preprocessor for large-scale inductive learning.

    This preprocessor processes samples one-by-one, applying transforms and
    immediately saving each to disk to maintain constant memory usage regardless
    of dataset size. This enables training on datasets that would otherwise cause
    out-of-memory errors during preprocessing/lifting operations.

    The dataset supports transform caching via parameter hashing, ensuring that
    identical transform configurations reuse previously processed data.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
        Source dataset to process. Can be any PyG dataset or PyTorch dataset.
    data_dir : str or Path
        Root directory for storing processed samples.
    transforms_config : DictConfig, optional
        Configuration parameters for transforms (liftings). If None, no
        transforms are applied and data is used as-is (default: None).
    force_reload : bool, optional
        If True, reprocess all samples even if cache exists (default: False).
    **kwargs : dict
        Additional arguments passed to parent Dataset class.

    Attributes
    ----------
    processed_dir : Path
        Directory containing processed sample files.
    num_samples : int
        Total number of samples in the dataset.
    transforms_parameters : dict
        Serialized transform parameters for cache validation.

    Examples
    --------
    >>> from torch_geometric.datasets import TUDataset
    >>> from omegaconf import DictConfig
    >>>
    >>> # Load source dataset
    >>> source = TUDataset(root='/tmp/data', name='ENZYMES')
    >>>
    >>> # Configure lifting transform
    >>> config = DictConfig({
    ...     'transform_name': 'liftings.graph2simplicial',
    ...     'complex_dim': 2
    ... })
    >>>
    >>> # Create on-disk dataset (processes sequentially)
    >>> dataset = OnDiskInductiveDataset(
    ...     dataset=source,
    ...     data_dir='/tmp/enzymes_processed',
    ...     transforms_config=config
    ... )
    >>>
    >>> # Use in training (lazy loading from disk)
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch in loader:
    ...     # Train model
    ...     pass

    Notes
    -----
    - Memory usage remains constant during processing (O(1) per sample)
    - Each sample is saved as an individual .pt file
    - Transform parameters are hashed to create unique cache directories
    - Compatible with existing TopoBench loaders and DataloadDataset
    - Processing progress is displayed via tqdm progress bar

    See Also
    --------
    topobench.data.preprocessor.preprocessor.PreProcessor :
        In-memory dataset preprocessor (original implementation).
    topobench.transforms.data_transform.DataTransform :
        Transform wrapper used for applying liftings.
    """

    def __init__(
        self,
        dataset: torch_geometric.data.Dataset | torch.utils.data.Dataset,
        data_dir: str | Path,
        transforms_config: DictConfig | None = None,
        force_reload: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize OnDiskInductiveDataset.

        Parameters
        ----------
        dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
            Source dataset to process.
        data_dir : str or Path
            Root directory for storing processed samples.
        transforms_config : DictConfig, optional
            Configuration parameters for transforms (default: None).
        force_reload : bool, optional
            If True, reprocess all samples even if cache exists (default: False).
        **kwargs : dict
            Additional arguments passed to parent Dataset class.
        """
        super().__init__()
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.force_reload = force_reload
        self.transforms_config = transforms_config

        # Store split_idx if available (for datasets with fixed splits)
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

        # Initialize transform and processed directory
        if transforms_config is not None:
            self.pre_transform = self._instantiate_pre_transform(
                transforms_config
            )
            self._set_processed_data_dir(transforms_config)
        else:
            # No transforms - use data as-is
            self.pre_transform = None
            self.processed_dir = self.data_dir / "no_transforms"

        # Ensure processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load or create metadata
        self.metadata_path = self.processed_dir / "metadata.json"
        if self._should_process():
            self._process_samples()
        else:
            self._load_metadata()

    def __repr__(self) -> str:
        """Return string representation of dataset.

        Returns
        -------
        str
            String describing the dataset and its size.
        """
        return (
            f"{self.__class__.__name__}("
            f"num_samples={self.num_samples}, "
            f"processed_dir={self.processed_dir})"
        )

    def __len__(self) -> int:
        """Return number of samples in dataset.

        Returns
        -------
        int
            Total number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        """Load sample from disk.

        Parameters
        ----------
        idx : int
            Sample index (0-indexed).

        Returns
        -------
        torch_geometric.data.Data
            Loaded data sample.

        Raises
        ------
        IndexError
            If index is out of range.
        FileNotFoundError
            If sample file doesn't exist on disk.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset of size "
                f"{self.num_samples}"
            )

        sample_path = self._get_sample_path(idx)

        if not sample_path.exists():
            raise FileNotFoundError(
                f"Sample file not found: {sample_path}. "
                f"Dataset may be corrupted. Try force_reload=True."
            )

        # Load sample from disk
        data = torch.load(sample_path)
        return data

    def _should_process(self) -> bool:
        """Check if dataset needs to be processed.

        Returns
        -------
        bool
            True if processing is needed, False if cache is valid.
        """
        if self.force_reload:
            return True

        if not self.metadata_path.exists():
            return True

        # Verify all sample files exist
        try:
            metadata = self._load_metadata_file()
            num_samples = metadata.get("num_samples", 0)

            for idx in range(num_samples):
                sample_path = self._get_sample_path(idx)
                if not sample_path.exists():
                    return True

            # Verify transform parameters match (if applicable)
            if self.transforms_config is not None:
                saved_params = metadata.get("transforms_parameters", {})
                if saved_params != self.transforms_parameters:
                    return True

            return False

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return True

    def _process_samples(self) -> None:
        """Iterate through samples, apply transforms, and save to disk.

        This method iterates through the source dataset, applies transforms
        to each sample individually, saves it to disk, and immediately clears
        it from memory. This ensures constant memory usage.
        """
        print(
            f"Processing {len(self.dataset)} samples to {self.processed_dir}"
        )

        # Clear existing files if force_reload
        if self.force_reload:
            self._clear_processed_files()

        # Process each sample sequentially
        for idx in tqdm(
            range(len(self.dataset)),
            desc="Processing samples",
            unit="sample",
        ):
            # Load single sample
            data = self.dataset[idx]

            # Apply transform if configured
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Save to disk
            sample_path = self._get_sample_path(idx)
            torch.save(data, sample_path)

            # Explicitly delete to free memory
            del data

        # Save metadata
        self.num_samples = len(self.dataset)
        self._save_metadata()

        print(f"✓ Processed {self.num_samples} samples successfully")

    def _instantiate_pre_transform(
        self, transforms_config: DictConfig
    ) -> torch_geometric.transforms.Compose:
        """Instantiate transform from configuration.

        Parameters
        ----------
        transforms_config : DictConfig
            Transform configuration parameters.

        Returns
        -------
        torch_geometric.transforms.Compose
            Composed transform object.
        """
        # Handle nested liftings config
        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings

        # Check if single or multiple transforms
        if "transform_name" in transforms_config:
            # Single transform
            pre_transforms_dict = {
                transforms_config.transform_name: DataTransform(
                    **transforms_config
                )
            }
        else:
            # Multiple transforms
            pre_transforms_dict = {
                key: DataTransform(**value)
                for key, value in transforms_config.items()
            }

        # Store parameters for caching
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        self.transforms_parameters = ensure_serializable(transforms_parameters)

        # Return composed transform
        return torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
        )

    def _set_processed_data_dir(self, transforms_config: DictConfig) -> None:
        """Set processed data directory based on transform parameters.

        Creates a unique directory path using parameter hashing to enable
        caching of preprocessed data across runs with identical configurations.

        Parameters
        ----------
        transforms_config : DictConfig
            Transform configuration parameters.
        """
        # Create repository name from transform keys
        repo_name = "_".join(list(transforms_config.keys()))

        # Hash transform parameters for unique cache directory
        params_hash = make_hash(self.transforms_parameters)

        # Set processed directory path
        self.processed_dir = self.data_dir / repo_name / f"{params_hash}"

    def _get_sample_path(self, idx: int) -> Path:
        """Get file path for sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Path
            Path to sample file on disk.
        """
        return self.processed_dir / f"sample_{idx:06d}.pt"

    def _clear_processed_files(self) -> None:
        """Clear all processed sample files."""
        for sample_file in self.processed_dir.glob("sample_*.pt"):
            sample_file.unlink()

        if self.metadata_path.exists():
            self.metadata_path.unlink()

    def _save_metadata(self) -> None:
        """Save dataset metadata to disk."""
        metadata = {
            "num_samples": self.num_samples,
            "source_dataset": str(type(self.dataset).__name__),
            "processed_dir": str(self.processed_dir),
        }

        # Add transform parameters if applicable
        if self.transforms_config is not None:
            metadata["transforms_parameters"] = self.transforms_parameters

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self) -> None:
        """Load dataset metadata from disk."""
        metadata = self._load_metadata_file()
        self.num_samples = metadata["num_samples"]

    def _load_metadata_file(self) -> dict:
        """Load and return metadata dictionary.

        Returns
        -------
        dict
            Metadata dictionary.

        Raises
        ------
        FileNotFoundError
            If metadata file doesn't exist.
        json.JSONDecodeError
            If metadata file is corrupted.
        """
        with open(self.metadata_path) as f:
            return json.load(f)

    def load_dataset_splits(
        self, split_params: DictConfig
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load dataset splits for training/validation/testing.

        This method creates DataloadDataset wrappers for use with TopoBench's
        TBDataloader. For inductive learning, separate datasets are created
        for each split.

        Parameters
        ----------
        split_params : DictConfig
            Parameters for splitting the dataset. Must include:
            - learning_setting: 'inductive' (required)
            - Additional split configuration (e.g., train_ratio, val_ratio)

        Returns
        -------
        tuple
            A tuple containing (train_dataset, val_dataset, test_dataset).
            Val and test may be None depending on split_params.

        Raises
        ------
        ValueError
            If learning_setting is not 'inductive' or is missing.

        Notes
        -----
        This is compatible with the existing TopoBench split loading utilities.
        The actual split logic is delegated to load_inductive_splits utility.
        """
        from topobench.data.utils import load_inductive_splits

        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting != "inductive":
            raise ValueError(
                f"OnDiskInductiveDataset only supports inductive learning. "
                f"Got: {split_params.learning_setting}"
            )

        # Create list view of dataset for split loading
        data_list = [self[i] for i in range(len(self))]
        self.data_list = data_list

        # Use existing split utility
        return load_inductive_splits(self, split_params)
