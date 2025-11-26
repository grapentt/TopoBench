"""Loader for ogbg-molpcba and mock molecular datasets."""

from omegaconf import DictConfig

from topobench.data.datasets.ogbg_molpcba import (
    MockMolecularDataset,
    OGBGMolPCBADataset,
)
from topobench.data.loaders.base import AbstractLoader


class OGBGMolPCBALoader(AbstractLoader):
    """OGB molecular property prediction dataset (437K graphs, 128 tasks)."""
    
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.subset_size = parameters.get("subset_size", None)
        self.split = parameters.get("split", "train")
        self.use_mock = parameters.get("use_mock", False)
        self.dataset = None
    
    def load_dataset(self):
        if self.dataset is None:
            if self.use_mock:
                self.dataset = MockMolecularDataset(
                    root=self.root_data_dir,
                    num_samples=self.subset_size or 100
                )
            else:
                self.dataset = OGBGMolPCBADataset(
                    root=self.root_data_dir,
                    split=self.split,
                    subset_size=self.subset_size
                )
        return self.dataset
    
    def load(self):
        dataset = self.load_dataset()
        return dataset, str(self.root_data_dir)
