# src/deepscratch/heads/projection.py

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepscratch.heads.base import Head


class ProjectionHead(Head):
    """
    Projection head for contrastive and other self-supervised objectives.

    Maps an encoder's embedding into a (typically smaller) space where a
    contrastive loss (e.g. InfoNCE / SimCLR-style) is applied. It is common
    practice to train with this head attached and then discard it at
    fine-tuning time, keeping only the underlying encoder. When
    ``normalize=True`` (the default), outputs are L2-normalized so that a
    dot product between two outputs is a cosine similarity — the standard
    setup for contrastive losses.

    Input shape:
        [batch_size, input_dim]

    Output shape:
        [batch_size, projection_dim]
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        hidden_dims: Sequence[int] = (),
        normalize: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=projection_dim,
        )

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")

        self.projection_dim = projection_dim
        self.hidden_dims = tuple(hidden_dims)
        self.normalize = normalize

        dims = [input_dim, *hidden_dims, projection_dim]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            layers.append(nn.Linear(in_dim, out_dim))

            is_last_layer = i == len(dims) - 2

            # No dropout here: dropout would inject noise directly into the
            # similarity computation the contrastive loss relies on.
            if not is_last_layer:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.net(embeddings)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x

    @property
    def task_type(self) -> str:
        return "projection"