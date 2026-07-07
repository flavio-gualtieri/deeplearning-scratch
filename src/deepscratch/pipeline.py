# src/deepscratch/pipeline.py

from typing import Any, Optional

import yaml
import torch
import torch.nn as nn

from deepscratch.encoders.base import Encoder
from deepscratch.heads.base import Head
from deepscratch.trainers.base import Trainer


class Pipeline(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        head: Head,
        trainer: Optional[Trainer] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.head = head
        self.trainer = trainer

        self._validate_components()

    def _validate_components(self) -> None:
        if self.encoder.embedding_dim != self.head.input_dim:
            raise ValueError(
                "Encoder and head dimension mismatch: "
                f"encoder.embedding_dim={self.encoder.embedding_dim}, "
                f"head.input_dim={self.head.input_dim}."
            )

    @property
    def input_modality(self) -> str:
        return self.encoder.input_modality

    @property
    def task_type(self) -> str:
        return self.head.task_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        outputs = self.head(embeddings)
        return outputs

    def fit(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        if self.trainer is None:
            raise RuntimeError(
                "No trainer attached to this pipeline. "
                "Pass a trainer when constructing the Pipeline."
            )

        return self.trainer.fit(
            model=self,
            train_loader=train_loader,
            val_loader=val_loader,
            **kwargs,
        )

    def evaluate(
        self,
        data_loader: Any,
        **kwargs: Any,
    ) -> Any:
        if self.trainer is None:
            raise RuntimeError(
                "No trainer attached to this pipeline. "
                "Pass a trainer when constructing the Pipeline."
            )

        return self.trainer.evaluate(
            model=self,
            data_loader=data_loader,
            **kwargs,
        )

    def predict(
        self,
        data_loader: Any,
        **kwargs: Any,
    ) -> Any:
        if self.trainer is None:
            raise RuntimeError(
                "No trainer attached to this pipeline. "
                "Pass a trainer when constructing the Pipeline."
            )

        return self.trainer.predict(
            model=self,
            data_loader=data_loader,
            **kwargs,
        )