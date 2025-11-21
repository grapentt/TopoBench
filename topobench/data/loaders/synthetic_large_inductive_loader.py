"""Loader for synthetic large inductive dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets.synthetic_large_inductive_dataset import (
    SyntheticLargeInductiveDataset,
)
from topobench.data.loaders.base import AbstractLoader


class SyntheticLargeInductiveLoader(AbstractLoader):
    """Loader for synthetic large inductive dataset.

    This loader handles loading or generating a synthetic dataset
    with many graphs designed to test memory limits.

    Parameters
    ----------
    parameters : DictConfig
        Configuration with the following parameters:
        - data_dir : str
            Root directory for the dataset
        - data_name : str
            Name of the dataset
        - num_graphs : int
            Number of graphs to generate
        - nodes_per_graph : int
            Average nodes per graph
        - degree : int
            Average degree per node
        - num_features : int
            Number of node features
        - num_classes : int
            Number of classes
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> SyntheticLargeInductiveDataset:
        """Load or generate the synthetic dataset.

        Returns
        -------
        SyntheticLargeInductiveDataset
            The loaded/generated dataset.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> SyntheticLargeInductiveDataset:
        """Initialize the synthetic dataset.

        Returns
        -------
        SyntheticLargeInductiveDataset
            The initialized dataset instance.
        """
        return SyntheticLargeInductiveDataset(
            root=str(self.root_data_dir),
            name=self.parameters.get("data_name", "SyntheticLargeInductive"),
            parameters=self.parameters,
        )

    def _redefine_data_dir(
        self, dataset: SyntheticLargeInductiveDataset
    ) -> Path:
        """Redefine the data directory based on the dataset configuration.

        Parameters
        ----------
        dataset : SyntheticLargeInductiveDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root
