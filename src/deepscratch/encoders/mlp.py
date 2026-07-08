# src/deepscratch/encoders/mlp.py

from typing import Sequence

import torch
import torch.nn as nn

from deepscratch.encoders.base import Encoder


class TabularMLPEncoder(Encoder):
    """
    A simple encoder for tabular data.

    It maps numerical feature columns into a learned embedding.

    Input shape:
        [batch_size, input_dim]

    Output shape:
        [batch_size, embedding_dim]
    """

    accepted_feature_types = {"numeric"}

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.0,
    ):
        super().__init__(embedding_dim=embedding_dim)

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_dim = input_dim

        dims = [input_dim, *hidden_dims, embedding_dim]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            layers.append(nn.Linear(in_dim, out_dim))

            is_last_layer = i == len(dims) - 2
            if not is_last_layer:
                layers.append(nn.ReLU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def input_modality(self) -> str:
        return "tabular"