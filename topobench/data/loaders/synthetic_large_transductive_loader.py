"""Loader for synthetic large transductive dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets.synthetic_large_transductive_dataset import (
    SyntheticLargeTransductiveDataset,
)
from topobench.data.loaders.base import AbstractLoader


class SyntheticLargeTransductiveLoader(AbstractLoader):
    """Loader for synthetic large transductive dataset.

    This loader handles loading or generating a synthetic large graph
    for transductive node classification, designed to test memory limits.

    Parameters
    ----------
    parameters : DictConfig
        Configuration with the following parameters:
        - data_dir : str
            Root directory for the dataset
        - data_name : str
            Name of the dataset
        - num_nodes : int
            Number of nodes in the graph
        - degree : int
            Average degree per node
        - num_features : int
            Number of node features
        - num_classes : int
            Number of classes for node classification
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> SyntheticLargeTransductiveDataset:
        """Load or generate the synthetic transductive dataset.

        Returns
        -------
        SyntheticLargeTransductiveDataset
            The loaded/generated dataset (containing single large graph).
        """
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> SyntheticLargeTransductiveDataset:
        """Initialize the synthetic transductive dataset.

        Returns
        -------
        SyntheticLargeTransductiveDataset
            The initialized dataset instance.
        """
        return SyntheticLargeTransductiveDataset(
            root=str(self.root_data_dir),
            name=self.parameters.get(
                "data_name", "SyntheticLargeTransductive"
            ),
            parameters=self.parameters,
        )

    def _redefine_data_dir(
        self, dataset: SyntheticLargeTransductiveDataset
    ) -> Path:
        """Redefine the data directory based on the dataset configuration.

        Parameters
        ----------
        dataset : SyntheticLargeTransductiveDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root
