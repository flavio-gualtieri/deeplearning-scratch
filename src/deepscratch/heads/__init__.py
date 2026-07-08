from deepscratch.heads.base import Head
from deepscratch.heads.classification import ClassificationHead
from deepscratch.heads.regression import RegressionHead
from deepscratch.heads.multilabel import MultiLabelClassificationHead
from deepscratch.heads.projection import ProjectionHead
from deepscratch.heads.sequence_tagging import SequenceTaggingHead

__all__ = [
    "Head",
    "ClassificationHead",
    "RegressionHead",
    "MultiLabelClassificationHead",
    "ProjectionHead",
    "SequenceTaggingHead",
]