# src/deepscratch/trainers/base.py

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch.nn as nn


class Trainer(ABC):

    @abstractmethod
    def fit(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        ...

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        data_loader: Any,
        **kwargs: Any,
    ) -> Any:
       ...

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        data_loader: Any,
        **kwargs: Any,
    ) -> Any:
        ...