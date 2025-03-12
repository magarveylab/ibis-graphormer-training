from typing import List, Optional

import numpy as np
import torch
from torch import nn

from omnicons.models import DataStructs, helpers


class SingleLabelNodeClassificationHead(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        num_labels: int = 2,
        class_weight: Optional[List[float]] = None,
        analyze_inputs: List[str] = ["a"],
        binary: bool = False,
        node_type: str = None,
        loss_scalar: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.node_type = node_type
        self.training_task = "node_classification"
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.binary = binary
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        if binary:
            if kwargs.get("multi_label", False) is True:
                self.loss_fct = nn.BCEWithLogitsLoss(weight=weight)
            else:
                if weight is not None and weight.ndim != 1:
                    raise ValueError(
                        f"Single-class binary labels should be passed as only the weight of the positive case, i.e. \
                                     1. Input expected is a single-value tensor, but you passed {weight.ndim}"
                    )
                self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            self.loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100, weight=weight
            )
        self.analyze_inputs = analyze_inputs
        self.loss_scalar = loss_scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def classification(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> DataStructs.ClassificationOutput:
        consider_index, labels = helpers.recast_labels_for_single_label(labels)
        x = helpers.recast_input_for_single_label(x, consider_index)
        logits = self.forward(x)
        if self.binary and labels.ndim == 1:
            labels = labels.view(
                labels.size(0), -1
            ).float()  # float labels required for BCEwithlogitsloss.
        loss = self.loss_fct(logits, labels)
        if torch.isnan(loss):
            loss = loss * self.loss_scalar
            # place holder values (if all values in batch is -100)
            # to ensure that there are no wholes in metrics
            new_labels = torch.LongTensor([0])
            logits = torch.Tensor([[1] + [0] * (self.num_labels - 1)])
            new_labels = new_labels.to(device=x.device)
            logits = logits.to(device=x.device)
        return {"logits": logits, "labels": labels, "loss": loss}


class MultiLabelNodeClassificationHead(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        num_labels: int = 2,
        class_weight: Optional[List[float]] = None,
        analyze_inputs: List[str] = ["a"],
        loss_scalar: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.training_task = "node_classification"
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        self.ignore_index = (
            -100
        )  # if the first element corresponds to -100, ignore in loss calc
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=weight)
        self.analyze_inputs = analyze_inputs
        self.loss_scalar = loss_scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def classification(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> DataStructs.ClassificationOutput:
        consider_index, labels = helpers.recast_labels_for_multi_label(labels)
        x = helpers.recast_input_for_multi_label(x, consider_index)
        logits = self.forward(x)
        loss = self.loss_fct(logits, labels)
        if torch.isnan(loss):
            loss = loss * self.loss_scalar
            # place holder values (if all values in batch is -100)
            # to ensure that there are no wholes in metrics
            new_labels = torch.LongTensor([0])
            logits = torch.Tensor([[1] + [0] * (self.num_labels - 1)])
            new_labels = new_labels.to(device=x.device)
            logits = logits.to(device=x.device)
        return {"logits": logits, "labels": labels, "loss": loss}
