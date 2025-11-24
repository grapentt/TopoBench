"""On-disk dataset class for US County Demographics dataset.

This implementation uses BaseOnDiskInductiveDataset to avoid loading the entire
dataset into RAM, making it suitable for memory-constrained environments.
"""

import os
import os.path as osp
import shutil
from pathlib import Path
from typing import ClassVar

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, extract_zip

from topobench.data.datasets.base_inductive import FileBasedInductiveDataset
from topobench.data.utils import (
    download_file_from_drive,
    read_us_county_demos,
)


class USCountyDemosOnDiskDataset(FileBasedInductiveDataset):
    r"""On-disk dataset class for US County Demographics.

    This dataset loads data on-demand to minimize memory usage, making it suitable
    for large-scale processing or memory-constrained environments.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset containing:
        - year: int, year of the data (default: 2012)
        - task_variable: str, prediction target variable

    Attributes
    ----------
    URLS : dict
        Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT : dict
        Dictionary containing the file formats for the dataset.

    Example
    -------
    >>> from omegaconf import DictConfig
    >>> params = DictConfig({"year": 2012, "task_variable": "Election"})
    >>> dataset = USCountyDemosOnDiskDataset(
    ...     root="/data",
    ...     name="US-county-demos",
    ...     parameters=params
    ... )
    >>> # Data is loaded on-demand
    >>> graph = dataset[0]  # Only loads when accessed
    """

    URLS: ClassVar = {
        "US-county-demos": "https://drive.google.com/file/d/1FNF_LbByhYNICPNdT6tMaJI9FxuSvvLK/view?usp=sharing",
    }

    FILE_FORMAT: ClassVar = {
        "US-county-demos": "zip",
    }

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
        cache_samples: bool = True,
    ) -> None:
        """Initialize the US County Demographics on-disk dataset.

        Parameters
        ----------
        root : str
            Root directory where the dataset will be saved.
        name : str
            Name of the dataset.
        parameters : DictConfig
            Configuration parameters for the dataset.
        cache_samples : bool, optional
            Enable caching of loaded samples (default: True).
        """
        # Store original arguments for pickling
        self._original_root = root
        self.name = name
        self.parameters = parameters
        self.year = parameters.year
        self.task_variable = parameters.task_variable

        # Set up directories
        self.dataset_root = osp.join(root, name)
        self.raw_dir = osp.join(self.dataset_root, "raw")
        self.processed_root = osp.join(
            self.dataset_root,
            "_".join([str(self.year), self.task_variable]),
        )
        self.processed_dir = osp.join(self.processed_root, "processed")

        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Download and process if needed
        if not self._raw_files_exist():
            self._download()

        if not self._processed_files_exist():
            self._process()

        # Set up file pattern for FileBasedInductiveDataset
        file_pattern = "data_*.pt"

        # Initialize parent class
        super().__init__(
            root=self.processed_dir,
            file_pattern=file_pattern,
            cache_samples=cache_samples,
        )

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"{self.__class__.__name__}("
            f"root={self.dataset_root}, "
            f"name={self.name}, "
            f"year={self.year}, "
            f"task_variable={self.task_variable}, "
            f"num_samples={len(self)})"
        )

    def _raw_files_exist(self) -> bool:
        """Check if raw files exist.

        Returns
        -------
        bool
            True if raw files exist, False otherwise.
        """
        raw_files = ["county_graph.csv", f"county_stats_{self.year}.csv"]
        return all(osp.exists(osp.join(self.raw_dir, f)) for f in raw_files)

    def _processed_files_exist(self) -> bool:
        """Check if processed files exist.

        Returns
        -------
        bool
            True if processed files exist, False otherwise.
        """
        processed_file = osp.join(self.processed_dir, "data_0.pt")
        return osp.exists(processed_file)

    def _download(self) -> None:
        """Download the dataset from a URL and save it to the raw directory."""
        print(f"Downloading {self.name}...")

        # Download data from the source
        url = self.URLS[self.name]
        file_format = self.FILE_FORMAT[self.name]
        download_file_from_drive(
            file_link=url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format=file_format,
        )

        # Extract zip file
        filename = f"{self.name}.{file_format}"
        path = osp.join(self.raw_dir, filename)
        extract_zip(path, self.raw_dir)

        # Delete zip file
        os.unlink(path)

        # Organize files: move files from subdirectory to raw_dir
        subdir = osp.join(self.raw_dir, self.name)
        if osp.exists(subdir):
            for file in os.listdir(subdir):
                shutil.move(osp.join(subdir, file), self.raw_dir)
            shutil.rmtree(subdir)

        print(f"Download complete: {self.name}")

    def _process(self) -> None:
        """Process the raw data and save it to disk."""
        print(f"Processing {self.name}...")

        # Load and process the data
        data = read_us_county_demos(
            self.raw_dir, self.year, self.task_variable
        )

        # Save as a single file (this dataset contains one graph)
        save_path = osp.join(self.processed_dir, "data_0.pt")
        torch.save(data, save_path)

        print(f"Processing complete: {self.name}")

    def _load_file(self, file_path: Path) -> Data:
        """Load a data sample from a file.

        Parameters
        ----------
        file_path : Path
            Path to the file to load.

        Returns
        -------
        Data
            The loaded PyTorch Geometric Data object.
        """
        return torch.load(file_path, weights_only=False)

    def _get_pickle_args(self) -> tuple:
        """Get arguments for pickling.

        Returns
        -------
        tuple
            Arguments to pass to __init__ during unpickling.
        """
        return (self._original_root, self.name, self.parameters, self.cache_samples)

    @property
    def num_features(self) -> int:
        """Return the number of node features.

        Returns
        -------
        int
            Number of node features.
        """
        # Load first sample to get feature dimension
        data = self[0]
        return data.x.shape[1] if data.x is not None else 0

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph.

        Returns
        -------
        int
            Number of nodes.
        """
        data = self[0]
        return data.num_nodes

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph.

        Returns
        -------
        int
            Number of edges.
        """
        data = self[0]
        return data.edge_index.shape[1] if data.edge_index is not None else 0
