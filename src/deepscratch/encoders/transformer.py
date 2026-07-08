# src/deepscratch/encoders/transformer.py

import math
from typing import Optional

import torch
import torch.nn as nn

from deepscratch.encoders.base import Encoder


class _SinusoidalPositionalEncoding(nn.Module):
    """Adds fixed (non-learned) sinusoidal positional information to a sequence."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape [1, max_len, dim] so it broadcasts over the batch dimension.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SequenceTransformerEncoder(Encoder):
    """
    A Transformer encoder for sequential data.

    It projects each sequence element into a model space, adds sinusoidal
    positional encodings, applies a stack of self-attention layers, and
    mean-pools across time into a learned embedding. When ``vocab_size`` is
    provided, integer token inputs are embedded first, so the same class
    handles both continuous sequences and discrete token sequences.

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
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
        vocab_size: Optional[int] = None,
    ):
        super().__init__(embedding_dim=embedding_dim)

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if model_dim <= 0 or model_dim % 2 != 0:
            raise ValueError("model_dim must be a positive even integer.")

        if num_heads <= 0 or model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")

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

        self.input_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = _SinusoidalPositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(model_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_dim = 2 if self.vocab_size is not None else 3
        if x.dim() != expected_dim:
            expected_shape = (
                "[batch, seq_len]" if self.vocab_size is not None else "[batch, seq_len, input_dim]"
            )
            raise ValueError(
                f"SequenceTransformerEncoder expects a {expected_dim}D input {expected_shape}, "
                f"got shape {tuple(x.shape)}. Without this check, a flat [batch, num_columns] "
                "MultiEncoder slice would silently broadcast against the positional encoding "
                "buffer instead of failing loudly whenever batch_size happens to equal "
                "model_dim -- this encoder needs a real sequence data pipeline, not tabular columns."
            )

        if self.token_embedding is not None:
            x = self.token_embedding(x)

        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)

        # Mean-pool across the sequence dimension to get one vector per sample.
        x = x.mean(dim=1)

        return self.head(x)

    @property
    def input_modality(self) -> str:
        return "sequence"