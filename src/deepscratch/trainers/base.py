# src/deepscratch/trainers/base.py

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch.nn as nn

from ..runs.context import RunContext
from ..runs.checkpointing import CheckpointManager
from ..runs.logging import JSONLLogger


class Trainer(ABC):
    def __init__(self, run_context: RunContext):
        self.run_context = run_context
        self.logger = JSONLLogger(
            path=run_context.path_for_log("events.jsonl"),
            run_id=run_context.run_id,
        )

        self.checkpointer = CheckpointManager(
            checkpoint_dir=run_context.checkpoints_dir,
            monitor=run_context.config.checkpointing.monitor,
            mode=run_context.config.checkpointing.mode,
            save_every=run_context.config.checkpointing.save_every,
            save_best=run_context.config.checkpointing.save_best,
        )

    @abstractmethod
    def fit(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        ...

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        data_loader: Any,
        **kwargs: Any,
    ) -> Any:
       ...

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        data_loader: Any,
        **kwargs: Any,
    ) -> Any:
        ...