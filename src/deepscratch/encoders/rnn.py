# src/deepscratch/encoders/rnn.py

from typing import Optional

import torch
import torch.nn as nn

from deepscratch.encoders.base import Encoder


class SequenceRNNEncoder(Encoder):
    """
    A recurrent encoder for sequential data.

    It processes a sequence with a (optionally bidirectional, optionally
    stacked) LSTM and projects the final hidden state into a learned
    embedding. When ``vocab_size`` is provided, integer token inputs are
    passed through an embedding table first, so the same class handles both
    continuous sequences and discrete token sequences.

    Input shape:
        [batch_size, seq_len, input_dim]   if vocab_size is None
        [batch_size, seq_len]              if vocab_size is set (long tensor of token ids)

    Output shape:
        [batch_size, embedding_dim]
    """

    accepted_feature_types = {"numeric"}

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        vocab_size: Optional[int] = None,
    ):
        super().__init__(embedding_dim=embedding_dim)

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        if vocab_size is not None and vocab_size <= 0:
            raise ValueError("vocab_size must be positive when provided.")

        self.input_dim = input_dim
        self.vocab_size = vocab_size

        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, input_dim)
        else:
            self.token_embedding = None

        # PyTorch only applies recurrent dropout *between* stacked layers, so
        # it has no effect (and warns) when num_layers == 1.
        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )

        num_directions = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_dim * num_directions, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_dim = 2 if self.vocab_size is not None else 3
        if x.dim() != expected_dim:
            expected_shape = (
                "[batch, seq_len]" if self.vocab_size is not None else "[batch, seq_len, input_dim]"
            )
            raise ValueError(
                f"SequenceRNNEncoder expects a {expected_dim}D input {expected_shape}, "
                f"got shape {tuple(x.shape)}. nn.LSTM silently treats a 2D input as an "
                "*unbatched* single sequence rather than raising a shape error, so a flat "
                "[batch, num_columns] MultiEncoder slice would otherwise corrupt training "
                "silently instead of failing loudly -- this encoder needs a real sequence "
                "data pipeline, not tabular columns."
            )

        if self.token_embedding is not None:
            x = self.token_embedding(x)

        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        _, (hidden, _) = self.rnn(x)

        if self.rnn.bidirectional:
            # The last layer's forward/backward states are the final two rows.
            last_forward = hidden[-2]
            last_backward = hidden[-1]
            final = torch.cat([last_forward, last_backward], dim=-1)
        else:
            final = hidden[-1]

        return self.head(final)

    @property
    def input_modality(self) -> str:
        return "sequence"