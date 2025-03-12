from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from omnicons.models import DataStructs, helpers


class SiameseGraphClassificationHead(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        num_labels: int = 2,
        class_weight: Optional[List[float]] = None,
        analyze_inputs: List[Tuple[str, str]] = [("a", "b")],
        **kwargs
    ):
        super().__init__()
        self.training_task = "siamese_graph_classification"
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.analyze_inputs = analyze_inputs

    def forward(
        self, pooled_output_a: torch.Tensor, pooled_output_b: torch.Tensor
    ) -> torch.Tensor:
        # new pooled output is squared difference
        pooled_output = torch.pow(pooled_output_a - pooled_output_b, 2)
        x = self.dropout(pooled_output)
        x = self.classifier(x)
        return x

    def classification(
        self,
        pooled_output_a: torch.Tensor,
        pooled_output_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> DataStructs.ClassificationOutput:
        consider_index, labels = helpers.recast_labels_for_single_label(labels)
        pooled_output_a = helpers.recast_input_for_single_label(
            pooled_output_a, consider_index
        )
        pooled_output_b = helpers.recast_input_for_single_label(
            pooled_output_b, consider_index
        )
        logits = self.forward(pooled_output_a, pooled_output_b)
        loss = self.loss_fct(logits, labels)
        return {"logits": logits, "labels": labels, "loss": loss}
