# entrypoint.py

import torch
import torch.nn as nn

from deepscratch.config import DeepScratchConfig
from deepscratch.data import DataModule, TabularCSVConfig, TabularCSVDataModule
from deepscratch.encoders import Encoder, TabularMLPEncoder
from deepscratch.heads import Head, ClassificationHead
from deepscratch.trainers import Trainer, BasicTorchTrainer
from deepscratch.runs.context import RunContext
from deepscratch.pipeline import Pipeline

LOSS_FNS = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


def build_data_module(config: DeepScratchConfig) -> DataModule:
    if config.data.type == "tabular_csv":
        return TabularCSVDataModule(
            TabularCSVConfig(
                path=config.general.input_path,
                feature_columns=config.data.feature_columns,
                target_column=config.data.target_column,
                task_type=config.general.task_type,
                batch_size=config.data.batch_size,
                splits=config.data.splits,
                shuffle=config.data.shuffle,
                random_seed=config.run.seed,
            )
        )

    raise ValueError(f"Unknown data type: {config.data.type}")


def build_encoder(config: DeepScratchConfig, data: DataModule) -> Encoder:
    encoder_config = config.components.encoder

    if encoder_config.type == "mlp":
        return TabularMLPEncoder(
            input_dim=data.input_dim,
            embedding_dim=encoder_config.embedding_dim,
            hidden_dims=encoder_config.hidden_dims,
            dropout=encoder_config.dropout,
        )

    raise ValueError(f"Unknown encoder type: {encoder_config.type}")


def build_head(config: DeepScratchConfig, data: DataModule) -> Head:
    head_config = config.components.head

    if head_config.type == "classification":
        return ClassificationHead(
            input_dim=head_config.input_dim,
            num_classes=data.output_dim,
            hidden_dims=head_config.hidden_dims,
            dropout=head_config.dropout,
        )

    raise ValueError(f"Unknown head type: {head_config.type}")


def build_trainer(config: DeepScratchConfig, run_context: RunContext) -> Trainer:
    trainer_config = config.components.trainer

    if trainer_config.type == "basic":
        if trainer_config.loss_fn not in LOSS_FNS:
            raise ValueError(f"Unknown loss_fn: {trainer_config.loss_fn}")

        if trainer_config.optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {trainer_config.optimizer}")

        return BasicTorchTrainer(
            epochs=trainer_config.epochs,
            learning_rate=trainer_config.learning_rate,
            loss_fn=LOSS_FNS[trainer_config.loss_fn](),
            run_context=run_context,
            optimizer_cls=OPTIMIZERS[trainer_config.optimizer],
            device=trainer_config.device,
        )

    raise ValueError(f"Unknown trainer type: {trainer_config.type}")


config = DeepScratchConfig.from_yaml("config.yaml")

run_context = RunContext(
    config=config,
    config_path="config.yaml",
)

run_context.setup()

data = build_data_module(config)
data.setup()

saved_data_artifacts = data.save_artifacts(run_context)

encoder = build_encoder(config, data=data)
head = build_head(config, data=data)
trainer = build_trainer(config, run_context=run_context)

pipeline = Pipeline(
    encoder=encoder,
    head=head,
    trainer=trainer,
)

history = pipeline.fit(
    train_loader=data.train_dataloader(),
    val_loader=data.val_dataloader(),
    run_context=run_context,
)
