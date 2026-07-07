# src/deepscratch/runs/context.py

import json
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from deepscratch.config import DeepScratchConfig


class RunContext:

    def __init__(
        self,
        config: DeepScratchConfig,
        config_path: Optional[str | Path] = None,
    ):
        self.config = config
        self.config_path = Path(config_path) if config_path is not None else None

        self.run_id = self._create_run_id(config.run.name)

        self.output_root = Path(config.general.output_path)
        self.run_dir = self.output_root / self.run_id

        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.metrics_dir = self.run_dir / "metrics"

    def setup(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.logs_dir.mkdir(parents=True)
        self.checkpoints_dir.mkdir(parents=True)
        self.artifacts_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)

        self.save_config()
        self.save_metadata()

    def save_config(self) -> None:
        frozen_config_path = self.run_dir / "config.json"

        with open(frozen_config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        if self.config_path is not None:
            shutil.copy2(self.config_path, self.run_dir / "config.yaml")

    def save_metadata(self) -> None:
        metadata = {
            "run_id": self.run_id,
            "created_at": datetime.utcnow().isoformat(),
            "task_type": self.config.general.task_type,
            "input_path": self.config.general.input_path,
        }

        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def path_for_checkpoint(self, name: str) -> Path:
        return self.checkpoints_dir / name

    def path_for_artifact(self, name: str) -> Path:
        return self.artifacts_dir / name

    def path_for_log(self, name: str) -> Path:
        return self.logs_dir / name

    def path_for_metrics(self, name: str) -> Path:
        return self.metrics_dir / name

    def _create_run_id(self, run_name: Optional[str]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_code = uuid.uuid4().hex[:8]

        if run_name:
            safe_name = run_name.lower().replace(" ", "-")
            return f"{timestamp}-{safe_name}-{short_code}"

        return f"{timestamp}-{short_code}"