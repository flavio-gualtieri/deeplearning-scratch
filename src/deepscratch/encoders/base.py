# src/deepscratch/encoders/base.py

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn as nn


class Encoder(nn.Module, ABC):
    accepted_feature_types: ClassVar[set[str]]

    def __init__(self, embedding_dim: int):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def input_modality(self) -> str:
        ...