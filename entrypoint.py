from typing import Optional

import torch
import torch.nn as nn

from deepscratch.config import DeepScratchConfig
from deepscratch.data import DataModule, TabularCSVConfig, TabularCSVDataModule
from deepscratch.encoders import (
    Encoder,
    TabularMLPEncoder,
    CategoricalEmbeddingEncoder,
    ImageCNNEncoder,
    SequenceRNNEncoder,
    SequenceTransformerEncoder,
)
from deepscratch.encoders.multiencoder import MultiEncoder
from deepscratch.heads import (
    Head,
    ClassificationHead,
    RegressionHead,
    MultiLabelClassificationHead,
    ProjectionHead,
    SequenceTaggingHead,
)
from deepscratch.trainers import Trainer, BasicTorchTrainer
from deepscratch.runs.context import RunContext
from deepscratch.pipeline import Pipeline

LOSS_FNS = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.SmoothL1Loss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
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


def build_leaf_encoder(
    encoder_config,
    input_dim: int,
    cardinalities: Optional[list[int]] = None,
) -> Encoder:
    """Build a single concrete encoder for one MultiEncoder group."""
    if encoder_config.type == "mlp":
        return TabularMLPEncoder(
            input_dim=input_dim,
            embedding_dim=encoder_config.embedding_dim,
            hidden_dims=encoder_config.hidden_dims,
            dropout=encoder_config.dropout,
        )

    if encoder_config.type == "categorical":
        if not cardinalities:
            raise ValueError(
                "A categorical encoder group needs a cardinality for every "
                "column in it, derived from the training data."
            )

        return CategoricalEmbeddingEncoder(
            cardinalities=cardinalities,
            embedding_dim=encoder_config.embedding_dim,
            dropout=encoder_config.dropout,
        )

    if encoder_config.type == "cnn":
        return ImageCNNEncoder(
            in_channels=input_dim,
            embedding_dim=encoder_config.embedding_dim,
            channels=encoder_config.channels or (32, 64, 128),
            kernel_size=encoder_config.kernel_size if encoder_config.kernel_size is not None else 3,
            dropout=encoder_config.dropout,
        )

    if encoder_config.type == "rnn":
        return SequenceRNNEncoder(
            input_dim=input_dim,
            embedding_dim=encoder_config.embedding_dim,
            hidden_dim=encoder_config.hidden_dim if encoder_config.hidden_dim is not None else 128,
            num_layers=encoder_config.num_layers if encoder_config.num_layers is not None else 1,
            bidirectional=bool(encoder_config.bidirectional),
            dropout=encoder_config.dropout,
        )

    if encoder_config.type == "transformer":
        return SequenceTransformerEncoder(
            input_dim=input_dim,
            embedding_dim=encoder_config.embedding_dim,
            model_dim=encoder_config.model_dim if encoder_config.model_dim is not None else 128,
            num_heads=encoder_config.num_heads if encoder_config.num_heads is not None else 4,
            num_layers=encoder_config.num_layers if encoder_config.num_layers is not None else 2,
            feedforward_dim=(
                encoder_config.feedforward_dim if encoder_config.feedforward_dim is not None else 256
            ),
            dropout=encoder_config.dropout,
        )

    raise ValueError(f"Unknown encoder type: {encoder_config.type}")


def infer_feature_type_for_group(group_config, encoder: Encoder) -> str:
    """
    MultiEncoder validates that the feature type routed to each leaf encoder is
    accepted by that encoder. If the config eventually grows a group-level
    feature_type field, use it; otherwise infer it when the encoder accepts only
    one feature type.
    """
    explicit_feature_type = getattr(group_config, "feature_type", None)
    if explicit_feature_type is not None:
        return explicit_feature_type

    accepted_feature_types = getattr(encoder, "accepted_feature_types", None)
    if not accepted_feature_types:
        raise ValueError(
            f"Cannot infer feature_type for encoder {type(encoder).__name__}: "
            "accepted_feature_types is empty or undefined."
        )

    if len(accepted_feature_types) == 1:
        return next(iter(accepted_feature_types))

    raise ValueError(
        f"Cannot infer feature_type for encoder {type(encoder).__name__}. "
        f"It accepts multiple feature types: {sorted(accepted_feature_types)}. "
        "Add a feature_type field to the encoder group config."
    )


def build_encoder(config: DeepScratchConfig, data: DataModule) -> Encoder:
    encoder_config = config.components.encoder

    column_to_index = {
        column_name: index
        for index, column_name in enumerate(config.data.feature_columns)
    }

    encoders: list[Encoder] = []
    features: list[list[int]] = []
    input_dims: list[int] = []
    embedding_dims: list[int] = []
    feature_types: list[str] = []

    for group_index, group_config in enumerate(encoder_config.groups):
        missing_columns = [
            column_name
            for column_name in group_config.feature_columns
            if column_name not in column_to_index
        ]
        if missing_columns:
            raise ValueError(
                f"Encoder group {group_index} references columns that are not in "
                f"data.feature_columns: {missing_columns}."
            )

        group_feature_indices = [
            column_to_index[column_name]
            for column_name in group_config.feature_columns
        ]
        group_input_dim = len(group_feature_indices)

        cardinalities = None
        if group_config.encoder.type == "categorical":
            data_cardinalities = getattr(data, "feature_cardinalities", None)
            if not data_cardinalities:
                raise ValueError(
                    f"Encoder group {group_index} is categorical, but the data "
                    "module does not expose feature_cardinalities (e.g. "
                    "TabularCSVDataModule after setup())."
                )

            cardinalities = [
                data_cardinalities[column_name]
                for column_name in group_config.feature_columns
            ]

        encoder = build_leaf_encoder(
            encoder_config=group_config.encoder,
            input_dim=group_input_dim,
            cardinalities=cardinalities,
        )

        encoders.append(encoder)
        features.append(group_feature_indices)
        input_dims.append(group_input_dim)
        embedding_dims.append(encoder.embedding_dim)
        feature_types.append(
            infer_feature_type_for_group(
                group_config=group_config,
                encoder=encoder,
            )
        )

    return MultiEncoder(
        encoders=encoders,
        features=features,
        input_dims=input_dims,
        embedding_dims=embedding_dims,
        feature_types=feature_types,
    )


def build_head(config: DeepScratchConfig, data: DataModule) -> Head:
    head_config = config.components.head

    if head_config.type == "classification":
        return ClassificationHead(
            input_dim=head_config.input_dim,
            num_classes=data.output_dim,
            hidden_dims=head_config.hidden_dims,
            dropout=head_config.dropout,
        )

    if head_config.type == "regression":
        return RegressionHead(
            input_dim=head_config.input_dim,
            output_dim=head_config.output_dim,
            hidden_dims=head_config.hidden_dims,
            dropout=head_config.dropout,
        )

    if head_config.type == "multilabel":
        return MultiLabelClassificationHead(
            input_dim=head_config.input_dim,
            num_labels=head_config.output_dim,
            hidden_dims=head_config.hidden_dims,
            dropout=head_config.dropout,
        )

    if head_config.type == "projection":
        return ProjectionHead(
            input_dim=head_config.input_dim,
            projection_dim=head_config.output_dim,
            hidden_dims=head_config.hidden_dims,
            normalize=head_config.normalize,
        )

    if head_config.type == "sequence_tagging":
        return SequenceTaggingHead(
            input_dim=head_config.input_dim,
            num_tags=head_config.output_dim,
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
