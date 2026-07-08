# src/deepscratch/heads/multilabel.py

from collections.abc import Sequence

import torch
import torch.nn as nn

from deepscratch.heads.base import Head


class MultiLabelClassificationHead(Head):
    """
    Multi-label classification head with optional hidden layers.

    Unlike ``ClassificationHead``, labels here are not mutually exclusive:
    any number of the ``num_labels`` classes can be active at once (e.g.
    tagging an image with multiple attributes, or assigning several topics
    to a document). Returns raw logits — pair with ``nn.BCEWithLogitsLoss``
    during training and apply ``torch.sigmoid`` (not softmax) at inference
    time to recover independent per-label probabilities.

    Input shape:
        [batch_size, input_dim]

    Output shape:
        [batch_size, num_labels]
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=num_labels,
        )

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if num_labels <= 0:
            raise ValueError("num_labels must be positive.")

        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")

        self.num_labels = num_labels
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

        dims = [input_dim, *hidden_dims, num_labels]
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
        return "multilabel_classification"