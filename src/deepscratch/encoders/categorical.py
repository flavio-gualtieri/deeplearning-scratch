# src/deepscratch/encoders/categorical.py

from typing import Optional, Sequence

import torch
import torch.nn as nn

from deepscratch.encoders.base import Encoder


class CategoricalEmbeddingEncoder(Encoder):
    """
    An encoder for categorical tabular data.

    Each categorical feature (column) gets its own embedding table. The
    per-feature embeddings are concatenated and projected into a shared
    embedding. This pairs naturally with ``TabularMLPEncoder``, which handles
    the numerical columns.

    Input shape:
        [batch_size, num_features]   (long tensor of category indices per column)

    Output shape:
        [batch_size, embedding_dim]
    """

    accepted_feature_types = {"categorical"}

    def __init__(
        self,
        cardinalities: Sequence[int],
        embedding_dim: int,
        feature_embedding_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__(embedding_dim=embedding_dim)

        if len(cardinalities) == 0:
            raise ValueError("cardinalities must contain at least one value.")

        if any(c <= 0 for c in cardinalities):
            raise ValueError("every cardinality must be positive.")

        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        if feature_embedding_dims is None:
            # Common heuristic: size each table relative to its cardinality,
            # capped at 50 so high-cardinality columns stay manageable.
            feature_embedding_dims = [min(50, (c + 1) // 2) for c in cardinalities]

        if len(feature_embedding_dims) != len(cardinalities):
            raise ValueError(
                "feature_embedding_dims must have one entry per cardinality."
            )

        if any(d <= 0 for d in feature_embedding_dims):
            raise ValueError("every feature embedding dim must be positive.")

        self.num_features = len(cardinalities)

        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dim)
            for cardinality, dim in zip(cardinalities, feature_embedding_dims)
        )

        total_dim = sum(feature_embedding_dims)

        head: list[nn.Module] = []
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(total_dim, embedding_dim))

        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_features] -> embed each column, then concatenate.
        # Upstream data loading stores category codes in a float32 tensor
        # alongside numeric columns, so cast back to the integer indices
        # nn.Embedding requires.
        embedded = [
            embedding(x[:, i].long()) for i, embedding in enumerate(self.embeddings)
        ]
        x = torch.cat(embedded, dim=-1)
        return self.head(x)

    @property
    def input_modality(self) -> str:
        return "tabular"