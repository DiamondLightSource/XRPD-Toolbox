from functools import cached_property
from pathlib import Path

import numpy as np
from h5py import Dataset, File

from xrpd_toolbox.utils.utils import get_entry, h5_to_array


class BaseDataLoader:
    """
    Base class for detector data loaders.
    Handles Nexus/HDF5 access and metadata retrieval.
    """

    def __init__(self, filepath: str | Path, data_path: str):
        self.filepath = Path(filepath)
        self.data_path = data_path

        self.entry = get_entry(self.filepath)
        self.dataset_path = f"/{self.entry}/{self.data_path}/data"

    def get_entries(self):
        paths = []

        with File(self.filepath, "r") as file:
            file.visit(paths.append)

        return paths

    def _get_dataset(self, dataset_path: str) -> Dataset:
        with File(self.filepath, "r") as file:
            if dataset_path not in file:
                raise ValueError(
                    f"Dataset path {dataset_path} not found in {self.filepath}"
                )

            data = file.get(dataset_path)

            if not isinstance(data, Dataset):
                raise ValueError(f"{dataset_path} is not a dataset")

            return data

    def get_data(self, dataset_path: str | None = None, selection=...) -> np.ndarray:
        dataset_path = dataset_path or self.dataset_path

        with File(self.filepath, "r") as file:
            if dataset_path not in file:
                raise ValueError(
                    f"Dataset path {dataset_path} not found in {self.filepath}"
                )

            data = file.get(dataset_path)

            if data is None or not isinstance(data, Dataset):
                raise ValueError(f"Data at {dataset_path} in {self.filepath} is None.")

            if data.ndim < 1:
                raise ValueError("Data has insufficient dimensions.")

            return np.asarray(data[selection])

    @property
    def data(self) -> np.ndarray:
        """Load the entire dataset."""
        return self.get_data()

    @cached_property
    def durations(self) -> np.ndarray:
        path = f"/{self.entry}/instrument/{self.data_path}/count_time"
        return h5_to_array(self.filepath, path)

    def read_array(self, path: str) -> np.ndarray:
        """Helper for reading arbitrary datasets."""
        return h5_to_array(self.filepath, path)
