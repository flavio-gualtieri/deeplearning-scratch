# src/deepscratch/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
    type: str
    input_dim: int
    embedding_dim: int
    hidden_dims: list[int]
    dropout: float = 0.0


@dataclass
class HeadConfig:
    type: str
    input_dim: int
    output_dim: int
    hidden_dims: list[int]
    dropout: float = 0.0


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
class ComponentsConfig:
    encoder: EncoderConfig
    head: HeadConfig
    trainer: TrainerConfig


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

        config = cls(
            general=GeneralConfig(**raw["general"]),
            run=RunConfig(**raw.get("run", {})),
            data=DataConfig(**raw["data"]),
            components=ComponentsConfig(
                encoder=EncoderConfig(**raw["components"]["encoder"]),
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