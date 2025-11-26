"""Custom loader for synthetic transductive dataset."""

import pickle
from pathlib import Path

import torch


class SyntheticTransductiveLoader:
    """Loader for synthetic transductive graph dataset.

    This loader handles large single-graph transductive learning datasets
    that are too large for standard in-memory approaches.
    """

    def __init__(self, config):
        """Initialize loader with config.

        Parameters
        ----------
        config : OmegaConf
            Configuration dict with data_dir parameter.
        """
        self.config = config
        self.data_dir = Path(config.data_dir)

    def load(self):
        """Load the dataset.

        Returns
        -------
        data : Data
            PyG Data object
        dataset_dir : str
            Path to dataset directory
        """
        data_file = self.data_dir / "data.pt"
        metadata_file = self.data_dir / "metadata.pkl"

        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_file}. "
                f"Please run create_synthetic_transductive_dataset.py first."
            )

        # Load data
        data = torch.load(data_file)

        # Load metadata
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            # Add metadata as attributes
            for key, value in metadata.items():
                if not hasattr(data, key):
                    setattr(data, key, value)

        return data, str(self.data_dir)
