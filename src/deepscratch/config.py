# src/deepscratch/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class GeneralConfig:
    input_path: str
    output_path: str
    task_type: str


@dataclass
class RunConfig:
    name: Optional[str] = None
    seed: int = 42


@dataclass
class DataConfig:
    type: str
    feature_columns: list[str]
    target_column: str
    batch_size: int
    splits: list[float]
    shuffle: bool = True


@dataclass
class EncoderConfig:
    # Leaf encoder config: mlp, cnn, categorical, rnn, transformer, etc.
    # Only a subset of these fields is meaningful for any given `type` --
    # each build_leaf_encoder() branch in entrypoint.py picks the ones it needs.
    type: str
    input_dim: Optional[int] = None
    embedding_dim: Optional[int] = None
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.0
    # cnn
    channels: Optional[list[int]] = None
    kernel_size: Optional[int] = None
    # rnn
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    bidirectional: Optional[bool] = None
    # transformer
    model_dim: Optional[int] = None
    num_heads: Optional[int] = None
    feedforward_dim: Optional[int] = None


@dataclass
class EncoderGroupConfig:
    feature_columns: list[str]
    encoder: EncoderConfig


@dataclass
class MultiEncoderConfig:
    # Always top-level encoder config.
    groups: list[EncoderGroupConfig]
    input_dim: Optional[int] = None
    embedding_dim: Optional[int] = None


@dataclass
class ComponentsConfig:
    encoder: MultiEncoderConfig
    head: HeadConfig
    trainer: TrainerConfig


@dataclass
class HeadConfig:
    type: str
    input_dim: int
    output_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.0
    normalize: bool = True  # only used by type="projection"


@dataclass
class TrainerConfig:
    type: str
    epochs: int
    learning_rate: float
    loss_fn: str
    optimizer: str
    device: Optional[str] = None
    metrics_every: int = 10


@dataclass
class LoggingConfig:
    type: str = "jsonl"
    log_every: int = 1


@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_every: int = 1
    save_best: bool = True
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class DeepScratchConfig:
    general: GeneralConfig
    run: RunConfig
    data: DataConfig
    components: ComponentsConfig
    logging: LoggingConfig
    checkpointing: CheckpointConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DeepScratchConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError("Config YAML is empty.")

        def build_encoder_group(raw_group: dict, index: int) -> EncoderGroupConfig:
            if "feature_columns" not in raw_group:
                raise ValueError(f"Encoder group {index} is missing feature_columns.")

            if "encoder" not in raw_group:
                raise ValueError(f"Encoder group {index} is missing encoder.")

            feature_columns = raw_group["feature_columns"]

            if not isinstance(feature_columns, list) or not feature_columns:
                raise ValueError(
                    f"Encoder group {index} feature_columns must be a non-empty list."
                )

            raw_encoder = raw_group["encoder"]
            encoder = EncoderConfig(**raw_encoder)

            expected_input_dim = len(feature_columns)

            if encoder.input_dim is None:
                encoder.input_dim = expected_input_dim
            elif encoder.input_dim != expected_input_dim:
                raise ValueError(
                    f"Encoder group {index} has {expected_input_dim} feature columns, "
                    f"but encoder.input_dim={encoder.input_dim}."
                )

            if encoder.embedding_dim is None:
                raise ValueError(
                    f"Encoder group {index} encoder must define embedding_dim."
                )

            return EncoderGroupConfig(
                feature_columns=feature_columns,
                encoder=encoder,
            )

        def build_multi_encoder_config(raw_encoder: dict) -> MultiEncoderConfig:
            raw_groups = raw_encoder.get("groups")

            if not isinstance(raw_groups, list) or not raw_groups:
                raise ValueError(
                    "components.encoder.groups must be a non-empty list."
                )

            groups = [
                build_encoder_group(raw_group, index)
                for index, raw_group in enumerate(raw_groups)
            ]

            seen_columns: set[str] = set()

            for group in groups:
                overlap = seen_columns.intersection(group.feature_columns)
                if overlap:
                    raise ValueError(
                        f"Feature columns {sorted(overlap)} are assigned to more than "
                        "one encoder group."
                    )

                seen_columns.update(group.feature_columns)

            total_input_dim = sum(group.encoder.input_dim for group in groups)
            total_embedding_dim = sum(group.encoder.embedding_dim for group in groups)

            input_dim = raw_encoder.get("input_dim")
            if input_dim is None:
                input_dim = total_input_dim
            elif input_dim != total_input_dim:
                raise ValueError(
                    f"components.encoder.input_dim={input_dim} does not match the sum "
                    f"of group input dimensions ({total_input_dim})."
                )

            embedding_dim = raw_encoder.get("embedding_dim")
            if embedding_dim is None:
                embedding_dim = total_embedding_dim
            elif embedding_dim != total_embedding_dim:
                raise ValueError(
                    f"components.encoder.embedding_dim={embedding_dim} does not match "
                    f"the sum of group embedding dimensions ({total_embedding_dim})."
                )

            return MultiEncoderConfig(
                groups=groups,
                input_dim=input_dim,
                embedding_dim=embedding_dim,
            )

        config = cls(
            general=GeneralConfig(**raw["general"]),
            run=RunConfig(**raw.get("run", {})),
            data=DataConfig(**raw["data"]),
            components=ComponentsConfig(
                encoder=build_multi_encoder_config(raw["components"]["encoder"]),
                head=HeadConfig(**raw["components"]["head"]),
                trainer=TrainerConfig(**raw["components"]["trainer"]),
            ),
            logging=LoggingConfig(**raw.get("logging", {})),
            checkpointing=CheckpointConfig(**raw.get("checkpointing", {})),
        )

        config.validate()
        return config

    def validate(self) -> None:
        if self.general.task_type not in {"classification", "regression"}:
            raise ValueError(
                "general.task_type must be either 'classification' or 'regression'."
            )

        if self.data.batch_size <= 0:
            raise ValueError("data.batch_size must be positive.")

        if len(self.data.splits) != 3:
            raise ValueError("data.splits must contain [train, val, test].")

        if abs(sum(self.data.splits) - 1.0) > 1e-6:
            raise ValueError("data.splits must sum to 1.0.")

        if self.components.encoder.embedding_dim != self.components.head.input_dim:
            raise ValueError(
                "Encoder embedding_dim must match head input_dim. "
                f"Got encoder.embedding_dim={self.components.encoder.embedding_dim}, "
                f"head.input_dim={self.components.head.input_dim}."
            )

        if self.components.trainer.epochs <= 0:
            raise ValueError("trainer.epochs must be positive.")

        if self.components.trainer.metrics_every <= 0:
            raise ValueError("trainer.metrics_every must be positive.")

        if self.checkpointing.save_every <= 0:
            raise ValueError("checkpointing.save_every must be positive.")

        if self.checkpointing.mode not in {"min", "max"}:
            raise ValueError("checkpointing.mode must be either 'min' or 'max'.")