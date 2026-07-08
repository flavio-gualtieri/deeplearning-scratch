# src/deepscratch/encoders/multiencoder.py

import torch
import torch.nn as nn

from deepscratch.encoders.base import Encoder


class MultiEncoder(Encoder):
    """
    Routes disjoint groups of input columns to their own sub-encoder, then
    concatenates the resulting embeddings.

    Input shape:
        [batch_size, total_input_dim]

    Output shape:
        [batch_size, sum(embedding_dim for each sub-encoder)]
    """

    def __init__(
        self,
        encoders: list[Encoder],
        features: list[list[int]],
        input_dims: list[int],
        embedding_dims: list[int],
        feature_types: list[str],
    ):
        self._validate_components(
            encoders, features, input_dims, embedding_dims, feature_types
        )

        super().__init__(embedding_dim=sum(embedding_dims))

        self.encoders = nn.ModuleList(encoders)
        self.features = features

    def _validate_components(
        self,
        encoders: list[Encoder],
        features: list[list[int]],
        input_dims: list[int],
        embedding_dims: list[int],
        feature_types: list[str],
    ) -> None:
        lengths = {
            "encoders": len(encoders),
            "features": len(features),
            "input_dims": len(input_dims),
            "embedding_dims": len(embedding_dims),
            "feature_types": len(feature_types),
        }

        if len(set(lengths.values())) > 1:
            raise ValueError(
                "encoders, features, input_dims, embedding_dims and feature_types "
                f"must all have the same length. Got: {lengths}."
            )

        if len(encoders) == 0:
            raise ValueError("MultiEncoder requires at least one encoder.")

        assigned_indices: set[int] = set()

        for group_indices in features:
            if len(group_indices) == 0:
                raise ValueError(
                    "Each feature group must contain at least one column index."
                )

            if any(index < 0 for index in group_indices):
                raise ValueError("Feature column indices must be non-negative.")

            overlap = assigned_indices.intersection(group_indices)
            if overlap:
                raise ValueError(
                    f"Column indices {sorted(overlap)} are assigned to more than "
                    "one encoder."
                )

            assigned_indices.update(group_indices)

        for i, (encoder, group_indices, input_dim, embedding_dim, feature_type) in enumerate(
            zip(encoders, features, input_dims, embedding_dims, feature_types)
        ):
            if not isinstance(encoder, Encoder):
                raise TypeError(
                    f"encoders[{i}] must be an Encoder instance, got {type(encoder)}."
                )

            if input_dim != len(group_indices):
                raise ValueError(
                    f"input_dims[{i}]={input_dim} does not match the number of "
                    f"column indices assigned to it ({len(group_indices)})."
                )

            if embedding_dim != encoder.embedding_dim:
                raise ValueError(
                    f"embedding_dims[{i}]={embedding_dim} does not match "
                    f"encoders[{i}].embedding_dim={encoder.embedding_dim}."
                )

            if feature_type not in encoder.accepted_feature_types:
                raise ValueError(
                    f"encoders[{i}] does not accept feature type '{feature_type}'. "
                    f"It accepts: {sorted(encoder.accepted_feature_types)}."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_embeddings = [
            encoder(x[:, group_indices])
            for encoder, group_indices in zip(self.encoders, self.features)
        ]

        return torch.cat(group_embeddings, dim=-1)

    @property
    def input_modality(self) -> str:
        modalities = {encoder.input_modality for encoder in self.encoders}

        if len(modalities) > 1:
            return "multi"

        return next(iter(modalities))
