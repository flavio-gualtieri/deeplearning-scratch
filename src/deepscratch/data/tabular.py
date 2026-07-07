# src/deepscratch/data/tabular.py

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from deepscratch.data.base import DataModule


@dataclass
class TabularCSVConfig:

    path: str
    feature_columns: list[str]
    target_column: str

    task_type: str = "classification"

    batch_size: int = 32
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    shuffle: bool = True
    random_seed: int = 42


class TabularDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)

        if targets.dtype.kind in {"i", "u"}:
            self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


class TabularCSVDataModule(DataModule):
    """
    DataModule for CSV files where the user selects feature columns
    and a target column.
    """

    def __init__(self, config: TabularCSVConfig):
        self.config = config

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None

        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None
        self.class_to_index: Optional[dict[str, int]] = None

    def setup(self) -> None:
        df = pd.read_csv(self.config.path)

        self._validate_columns(df)

        features = df[self.config.feature_columns].to_numpy(dtype=np.float32)
        raw_targets = df[self.config.target_column].to_numpy()

        if self.config.task_type == "classification":
            targets = self._encode_classification_targets(raw_targets)
            self._output_dim = len(self.class_to_index or {})
        elif self.config.task_type == "regression":
            targets = raw_targets.astype(np.float32)
            self._output_dim = 1
        else:
            raise ValueError(
                f"Unsupported task_type: {self.config.task_type}. "
                "Expected 'classification' or 'regression'."
            )

        self._input_dim = len(self.config.feature_columns)

        full_dataset = TabularDataset(
            features=features,
            targets=targets,
        )

        self._split_dataset(full_dataset)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing_features = [
            col for col in self.config.feature_columns if col not in df.columns
        ]

        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if self.config.target_column not in df.columns:
            raise ValueError(f"Missing target column: {self.config.target_column}")

    def _encode_classification_targets(self, raw_targets: np.ndarray) -> np.ndarray:
        unique_classes = sorted(set(raw_targets.tolist()))
        self.class_to_index = {
            class_name: index for index, class_name in enumerate(unique_classes)
        }

        encoded = np.array(
            [self.class_to_index[value] for value in raw_targets],
            dtype=np.int64,
        )

        return encoded

    def _split_dataset(self, dataset: Dataset) -> None:
        total_size = len(dataset)

        test_size = int(total_size * self.config.test_fraction)
        val_size = int(total_size * self.config.val_fraction)
        train_size = total_size - val_size - test_size

        if train_size <= 0:
            raise ValueError("Train split is empty. Reduce val/test fractions.")

        generator = torch.Generator().manual_seed(self.config.random_seed)

        splits = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )

        self._train_dataset = splits[0]
        self._val_dataset = splits[1] if val_size > 0 else None
        self._test_dataset = splits[2] if test_size > 0 else None

    def train_dataloader(self):
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")

        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
        )

    def val_dataloader(self):
        if self._val_dataset is None:
            return None

        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        if self._test_dataset is None:
            return None

        return DataLoader(
            self._test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            raise RuntimeError("Call setup() before accessing input_dim.")

        return self._input_dim

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("Call setup() before accessing output_dim.")

        return self._output_dim

    @property
    def input_modality(self) -> str:
        return "tabular"

    @property
    def task_type(self) -> str:
        return self.config.task_type