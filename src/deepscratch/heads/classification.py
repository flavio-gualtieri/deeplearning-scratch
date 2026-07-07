# src/deepscratch/heads/classification.py

from typing import Optional

import torch
import torch.nn as nn

from deepscratch.heads.base import Head


class ClassificationHead(Head):
    """
    A simple classification head.

    It maps encoder embeddings to class logits.

    Input shape:
        [batch_size, input_dim]

    Output shape:
        [batch_size, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=num_classes,
        )

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")

        self.num_classes = num_classes

        if hidden_dims is None:
            self.net = nn.Linear(input_dim, num_classes)
        else:
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, num_classes))
            self.net = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)

    @property
    def task_type(self) -> str:
        return "classification"