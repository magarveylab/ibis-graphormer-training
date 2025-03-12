from typing import Literal, Optional

from torch.nn import ModuleDict
from torchmetrics import Accuracy, F1Score, Precision, Recall


def get(
    name: str,
    num_labels: Optional[int] = None,
    num_classes: Optional[int] = None,
    task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
    average_strategy: Literal["micro", "macro", "weighted"] = "weighted",
    ignore_index: int = -100,
):
    # for multiclass define num_classes, for multilabel define num_labels
    metrics = ModuleDict()
    metrics[f"{name}_accuracy"] = Accuracy(
        task=task,
        num_classes=num_classes,
        num_labels=num_labels,
        average=average_strategy,
        ignore_index=ignore_index,
    )
    metrics[f"{name}_f1score"] = F1Score(
        task=task,
        num_classes=num_classes,
        num_labels=num_labels,
        average=average_strategy,
        ignore_index=ignore_index,
    )
    metrics[f"{name}_precision"] = Precision(
        task=task,
        num_classes=num_classes,
        num_labels=num_labels,
        average=average_strategy,
        ignore_index=ignore_index,
    )
    metrics[f"{name}_recall"] = Recall(
        task=task,
        num_classes=num_classes,
        num_labels=num_labels,
        average=average_strategy,
        ignore_index=ignore_index,
    )
    return metrics
