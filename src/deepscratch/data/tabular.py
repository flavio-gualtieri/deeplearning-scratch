# src/deepscratch/data/tabular.py

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset, Subset
from dataclasses import dataclass, field

from .base import DataModule
from ..runs.artifacts import save_json


@dataclass
class TabularCSVConfig:
    path: str
    feature_columns: list[str]
    target_column: str

    task_type: str = "classification"

    batch_size: int = 32
    splits: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
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

        self.num_rows: Optional[int] = None
        self.column_dtypes: Optional[dict[str, str]] = None
        self.split_indices: Optional[dict[str, list[int]]] = None

    def setup(self) -> None:
        df = pd.read_csv(self.config.path)

        self._validate_columns(df)

        self.num_rows = len(df)
        self.column_dtypes = {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        }

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

        train_fraction, val_fraction, test_fraction = self.config.splits

        train_size = int(total_size * train_fraction)
        val_size = int(total_size * val_fraction)
        test_size = total_size - train_size - val_size

        if train_size <= 0:
            raise ValueError("Train split is empty. Adjust data.splits.")

        if val_size < 0 or test_size < 0:
            raise ValueError("Invalid split sizes. Check data.splits.")

        generator = torch.Generator().manual_seed(self.config.random_seed)

        permutation = torch.randperm(
            total_size,
            generator=generator,
        ).tolist()

        train_indices = permutation[:train_size]
        val_indices = permutation[train_size : train_size + val_size]
        test_indices = permutation[train_size + val_size :]

        self.split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

        self._train_dataset = Subset(dataset, train_indices)
        self._val_dataset = (
            torch.utils.data.Subset(dataset, val_indices)
            if val_indices
            else None
        )
        self._test_dataset = (
            torch.utils.data.Subset(dataset, test_indices)
            if test_indices
            else None
        )

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
    
    def artifact_metadata(self) -> dict:
        if self.num_rows is None:
            raise RuntimeError("Call setup() before requesting artifact metadata.")

        return {
            "data_type": self.config.type if hasattr(self.config, "type") else "tabular_csv",
            "source_path": self.config.path,
            "input_modality": self.input_modality,
            "task_type": self.task_type,
            "num_rows": self.num_rows,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "feature_columns": self.config.feature_columns,
            "target_column": self.config.target_column,
            "batch_size": self.config.batch_size,
            "splits": self.config.splits,
            "shuffle": self.config.shuffle,
            "column_dtypes": self.column_dtypes,
        }


    def save_artifacts(self, run_context) -> list[str]:
        if self.split_indices is None:
            raise RuntimeError("Call setup() before saving data artifacts.")

        saved_paths = []

        data_schema_path = save_json(
            path=run_context.path_for_artifact("data_schema.json"),
            payload=self.artifact_metadata(),
        )
        saved_paths.append(str(data_schema_path))

        split_indices_path = save_json(
            path=run_context.path_for_artifact("split_indices.json"),
            payload=self.split_indices,
        )
        saved_paths.append(str(split_indices_path))

        if self.class_to_index is not None:
            class_to_index_path = save_json(
                path=run_context.path_for_artifact("class_to_index.json"),
                payload=self.class_to_index,
            )
            saved_paths.append(str(class_to_index_path))

        return saved_paths