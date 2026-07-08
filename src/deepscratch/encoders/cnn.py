# src/deepscratch/encoders/cnn.py

from typing import Sequence

import torch
import torch.nn as nn

from deepscratch.encoders.base import Encoder


class ImageCNNEncoder(Encoder):
    """
    A convolutional encoder for image data.

    It applies a stack of convolutional blocks (Conv -> BatchNorm -> ReLU ->
    MaxPool), globally pools the resulting feature maps, and projects them
    into a learned embedding. Global pooling means the encoder accepts images
    of varying spatial size without changing the output shape.

    Input shape:
        [batch_size, in_channels, height, width]

    Output shape:
        [batch_size, embedding_dim]
    """

    accepted_feature_types = {"numeric"}

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__(embedding_dim=embedding_dim)

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if len(channels) == 0:
            raise ValueError("channels must contain at least one value.")

        if any(c <= 0 for c in channels):
            raise ValueError("every channel width must be positive.")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.in_channels = in_channels

        dims = [in_channels, *channels]
        padding = kernel_size // 2
        blocks: list[nn.Module] = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            blocks.append(
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
            )
            blocks.append(nn.BatchNorm2d(out_dim))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(kernel_size=2))

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        head: list[nn.Module] = []
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(channels[-1], embedding_dim))

        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                "ImageCNNEncoder expects a 4D input [batch, in_channels, height, "
                f"width], got shape {tuple(x.shape)}. A MultiEncoder group only ever "
                "supplies flat [batch, num_columns] tabular slices, which are not "
                "image tensors -- this encoder needs a real image data pipeline."
            )

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)

    @property
    def input_modality(self) -> str:
        return "image"