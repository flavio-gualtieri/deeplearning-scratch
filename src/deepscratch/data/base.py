from abc import ABC, abstractmethod
from typing import Any


class DataModule(ABC):
    """
    Base abstraction for anything that prepares data.
    """

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self):
        raise NotImplementedError

    def artifact_metadata(self) -> dict[str, Any]:
        """
        Return metadata that should be saved with the run.

        Concrete DataModules can override this.
        """
        return {}

    def save_artifacts(self, run_context: Any) -> list[str]:
        """
        Save data-related artifacts for this run.

        Concrete DataModules can override this.

        Returns:
            List of saved file paths.
        """
        return []

    @property
    @abstractmethod
    def input_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_modality(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def task_type(self) -> str:
        raise NotImplementedError