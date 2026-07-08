# src/deepscratch/heads/sequence_tagging.py

from collections.abc import Sequence

import torch
import torch.nn as nn

from deepscratch.heads.base import Head


class SequenceTaggingHead(Head):
    """
    Per-timestep classification head for tasks like NER or POS tagging,
    where every element of a sequence gets its own label rather than the
    sequence as a whole getting one.

    The ``nn.Linear`` layers here apply along the last dimension only, so
    the same weights are shared across every timestep — this only differs
    from ``ClassificationHead`` in the shape it expects and returns.

    Note: this expects per-timestep embeddings (e.g. the output of a
    Transformer encoder *before* it mean-pools across time), not the
    pooled ``[batch_size, embedding_dim]`` vectors that this project's
    sequence encoders currently return. Pairing this head with
    ``SequenceRNNEncoder`` or ``SequenceTransformerEncoder`` as they
    stand today would need those encoders extended with a
    ``return_sequence`` option that skips the final pooling step.

    Input shape:
        [batch_size, seq_len, input_dim]

    Output shape:
        [batch_size, seq_len, num_tags]
    """

    def __init__(
        self,
        input_dim: int,
        num_tags: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=num_tags,
        )

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if num_tags <= 1:
            raise ValueError("num_tags must be greater than 1.")

        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")

        self.num_tags = num_tags
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

        dims = [input_dim, *hidden_dims, num_tags]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            # nn.Linear broadcasts over all leading dimensions, so this
            # applies independently and identically to every timestep.
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
        return "sequence_tagging"