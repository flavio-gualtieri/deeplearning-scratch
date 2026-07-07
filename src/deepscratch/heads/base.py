# src/deepscratch/heads/base.py

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class Head(nn.Module, ABC):

    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if output_dim is not None and output_dim <= 0:
            raise ValueError("output_dim must be positive if provided.")

        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def task_type(self) -> str:
        ...