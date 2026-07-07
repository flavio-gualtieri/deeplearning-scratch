# src/deepscratch/data/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DataSpec:
    """
    Describes the user's dataset choices.

    This is the kind of object that your UI/config layer can produce after
    the user uploads data and selects columns, labels, modalities, etc.
    """

    source: str
    modality: str

    feature_columns: Optional[list[str]] = None
    target_columns: Optional[list[str]] = None

    task_type: Optional[str] = None

    metadata: Optional[dict[str, Any]] = None


class DataModule(ABC):
    """
    Base abstraction for anything that prepares data.

    Concrete examples:
        TabularCSVDataModule
        ExcelDataModule
        ImageFolderDataModule
        TextCSVDataModule

    A DataModule owns:
        - reading the raw source
        - preprocessing
        - train/val/test splitting
        - creating PyTorch DataLoaders
    """

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def train_dataloader(self):
        ...

    @abstractmethod
    def val_dataloader(self):
        ...

    @abstractmethod
    def test_dataloader(self):
        ...

    @property
    @abstractmethod
    def input_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def input_modality(self) -> str:
        ...

    @property
    @abstractmethod
    def task_type(self) -> str:
        ...