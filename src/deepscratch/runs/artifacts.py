# src/deepscratch/runs/artifacts.py

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from .context import RunContext


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """
    Save a dictionary as pretty JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    return path


def save_final_model(
    run_context: RunContext,
    model: nn.Module,
    epoch: int,
    metrics: Optional[dict[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    history: Optional[Any] = None,
    filename: str = "final_model.pt",
) -> Path:
    """
    Save the final trained model artifact.

    Important design decision:
        We save the model state_dict, not the whole Python object.

    Why:
        Saving the whole model with torch.save(model) tightly couples the artifact
        to the exact Python class definitions and import paths.

        Saving state_dict + config is more portable.
    """

    path = run_context.path_for_artifact(filename)

    artifact = {
        "run_id": run_context.run_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "metrics": metrics or {},
        "model": {
            "module": model.__class__.__module__,
            "class_name": model.__class__.__name__,
            "state_dict": model.state_dict(),
        },
        "config": asdict(run_context.config),
        "history": _to_serializable(history),
    }

    if optimizer is not None:
        artifact["optimizer"] = {
            "module": optimizer.__class__.__module__,
            "class_name": optimizer.__class__.__name__,
            "state_dict": optimizer.state_dict(),
        }

    torch.save(artifact, path)

    return path


def save_model_summary(
    run_context: RunContext,
    model: nn.Module,
    filename: str = "model_summary.json",
) -> Path:
    """
    Save lightweight model metadata that can be inspected without loading PyTorch.
    """

    encoder = getattr(model, "encoder", None)
    head = getattr(model, "head", None)

    payload = {
        "run_id": run_context.run_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
        "input_modality": getattr(model, "input_modality", None),
        "task_type": getattr(model, "task_type", None),
        "encoder": None,
        "head": None,
        "num_parameters": count_parameters(model),
        "num_trainable_parameters": count_trainable_parameters(model),
    }

    if encoder is not None:
        payload["encoder"] = {
            "class_name": encoder.__class__.__name__,
            "module": encoder.__class__.__module__,
            "embedding_dim": getattr(encoder, "embedding_dim", None),
            "input_modality": getattr(encoder, "input_modality", None),
            "num_parameters": count_parameters(encoder),
            "num_trainable_parameters": count_trainable_parameters(encoder),
        }

    if head is not None:
        payload["head"] = {
            "class_name": head.__class__.__name__,
            "module": head.__class__.__module__,
            "input_dim": getattr(head, "input_dim", None),
            "output_dim": getattr(head, "output_dim", None),
            "task_type": getattr(head, "task_type", None),
            "num_parameters": count_parameters(head),
            "num_trainable_parameters": count_trainable_parameters(head),
        }

    return save_json(
        path=run_context.path_for_artifact(filename),
        payload=payload,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None

    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [_to_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    return str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return asdict(value)

    return str(value)