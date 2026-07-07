# src/deepscratch/runs/logging.py

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONLLogger:

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id

    def log_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self._write(
            {
                "type": "event",
                "event": event,
                "payload": payload or {},
            }
        )

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        split: str = "train",
    ) -> None:
        self._write(
            {
                "type": "metrics",
                "step": step,
                "split": split,
                "metrics": metrics,
            }
        )

    def log_error(self, error: BaseException) -> None:
        self._write(
            {
                "type": "error",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            **record,
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")