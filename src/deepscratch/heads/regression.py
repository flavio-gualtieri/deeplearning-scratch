# src/deepscratch/heads/regression.py

from collections.abc import Sequence

import torch
import torch.nn as nn

from deepscratch.heads.base import Head


class RegressionHead(Head):
    """
    Regression head with optional hidden layers.

    Predicts one or more continuous targets from a pooled embedding. Use
    ``output_dim=1`` (the default) for single-target regression (price,
    risk score, a forecast value) or ``output_dim > 1`` for multi-target
    regression (e.g. predicting several correlated measurements at once).

    Input shape:
        [batch_size, input_dim]

    Output shape:
        [batch_size, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
        )

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")

        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

        dims = [input_dim, *hidden_dims, output_dim]
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

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)

    @property
    def task_type(self) -> str:
        return "regression"