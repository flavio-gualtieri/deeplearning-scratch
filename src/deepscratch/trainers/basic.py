# src/deepscratch/trainers/basic.py

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import Trainer
from ..runs.context import RunContext
from ..runs.artifacts import save_final_model, save_model_summary


@dataclass
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]


class BasicTorchTrainer(Trainer):
    """
    A simple PyTorch trainer.

    This is intentionally minimal. It assumes each batch looks like:

        x, y = batch

    and that the model returns predictions/logits:

        y_hat = model(x)

    Good first use cases:
        - tabular classification
        - tabular regression
        - simple image classification
    """

    def __init__(
        self,
        epochs: int = 500,
        learning_rate: float = 1e-3,
        loss_fn: Optional[nn.Module] = None,
        run_context: RunContext = RunContext,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        device: Optional[str] = None,
    ):
        super().__init__(
            run_context=run_context
        )
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.device = torch.device(device or self._default_device())

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> TrainingHistory:
        model.to(self.device)

        loss_fn = self.loss_fn or self._default_loss_fn(model)

        optimizer = self.optimizer_cls(
            model.parameters(),
            lr=self.learning_rate,
        )

        history = TrainingHistory(
            train_losses=[],
            val_losses=[],
        )
        metrics_every = self.run_context.config.components.trainer.metrics_every
        last_metrics: dict[str, float] = {}

        try:
            self.logger.log_event("training_started")

            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_one_epoch(
                    model=model,
                    train_loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                )

                history.train_losses.append(train_loss)

                should_compute_metrics = epoch % metrics_every == 0

                if should_compute_metrics:
                    last_metrics = {
                        "train_loss": train_loss,
                    }

                    if val_loader is not None:
                        val_loss = self._evaluate_loss(
                            model=model,
                            data_loader=val_loader,
                            loss_fn=loss_fn,
                        )

                        history.val_losses.append(val_loss)
                        last_metrics["val_loss"] = val_loss

                    self.logger.log_metrics(
                        step=epoch,
                        split="train",
                        metrics={"train_loss": train_loss},
                    )

                    if "val_loss" in last_metrics:
                        self.logger.log_metrics(
                            step=epoch,
                            split="val",
                            metrics={"val_loss": last_metrics["val_loss"]},
                        )

                    saved_paths = self.checkpointer.maybe_save(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        metrics=last_metrics,
                    )

                    for path in saved_paths:
                        self.logger.log_event(
                            "checkpoint_saved",
                            {"path": str(path), "epoch": epoch},
                        )

                if "val_loss" in last_metrics:
                    print(
                        f"Epoch {epoch}/{self.epochs} "
                        f"- train_loss: {train_loss:.4f} "
                        f"- val_loss: {last_metrics['val_loss']:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{self.epochs} "
                        f"- train_loss: {train_loss:.4f}"
                    )

            final_model_path = save_final_model(
                run_context=self.run_context,
                model=model,
                epoch=self.epochs,
                metrics=last_metrics,
                optimizer=optimizer,
                history=history,
            )

            model_summary_path = save_model_summary(
                run_context=self.run_context,
                model=model,
            )

            self.logger.log_event(
                "final_model_saved",
                {
                    "path": str(final_model_path),
                    "epoch": self.epochs,
                },
            )

            self.logger.log_event(
                "model_summary_saved",
                {
                    "path": str(model_summary_path),
                },
            )

            self.logger.log_event("training_completed")

            return history

        except Exception as error:
            self.logger.log_error(error)
            raise

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        **kwargs: Any,
    ) -> dict[str, float]:
        model.to(self.device)

        loss_fn = self.loss_fn or self._default_loss_fn(model)

        loss = self._evaluate_loss(
            model=model,
            data_loader=data_loader,
            loss_fn=loss_fn,
        )

        return {
            "loss": loss,
        }

    def predict(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        **kwargs: Any,
    ) -> torch.Tensor:
        model.to(self.device)
        model.eval()

        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                x = self._get_inputs(batch)
                x = x.to(self.device)

                outputs = model(x)
                predictions.append(outputs.cpu())

        return torch.cat(predictions, dim=0)

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()

        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            x, y = self._unpack_batch(batch)

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / total_examples

    def _evaluate_loss(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> float:
        model.eval()

        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in data_loader:
                x, y = self._unpack_batch(batch)

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = model(x)
                loss = loss_fn(outputs, y)

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

        return total_loss / total_examples

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError(
                "BasicTorchTrainer expects each batch to be a tuple of "
                "(inputs, targets)."
            )

        x, y = batch
        return x, y

    def _get_inputs(self, batch: Any) -> torch.Tensor:
        """
        Used during prediction.

        Supports either:
            x
        or:
            x, y
        """

        if isinstance(batch, (tuple, list)):
            return batch[0]

        return batch

    def _default_loss_fn(self, model: nn.Module) -> nn.Module:
        """
        Pick a basic loss function based on the pipeline's task_type.
        """

        task_type = getattr(model, "task_type", None)

        if task_type == "classification":
            return nn.CrossEntropyLoss()

        if task_type == "regression":
            return nn.MSELoss()

        raise ValueError(
            "No loss_fn was provided, and the trainer could not infer one. "
            "Pass loss_fn manually or make sure the model has a valid task_type."
        )

    def _default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"