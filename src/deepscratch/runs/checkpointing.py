# src/deepscratch/runs/checkpointing.py

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


class CheckpointManager:

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_every: int = 1,
        save_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_every = save_every
        self.save_best = save_best

        self.best_value: Optional[float] = None

        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")

    def maybe_save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, float],
        extra: Optional[dict[str, Any]] = None,
    ) -> list[Path]:
        saved_paths: list[Path] = []

        if epoch % self.save_every == 0:
            path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            self._save_checkpoint(
                path=path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                extra=extra,
            )
            saved_paths.append(path)

        if self.save_best and self.monitor in metrics:
            current_value = metrics[self.monitor]

            if self._is_best(current_value):
                self.best_value = current_value
                path = self.checkpoint_dir / "best.pt"
                self._save_checkpoint(
                    path=path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    extra=extra,
                )
                saved_paths.append(path)

        return saved_paths

    def _is_best(self, value: float) -> bool:
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < self.best_value

        return value > self.best_value

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, float],
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "extra": extra or {},
        }

        torch.save(checkpoint, path)