from typing import List, Optional, Tuple, Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.heads.NodeClassification import (
    MultiLabelNodeClassificationHead,
    SingleLabelNodeClassificationHead,
)
from omnicons.models.heads.SiameseClassification import (
    SiameseGraphClassificationHead,
)


class NodeClsTaskHeadConfig(ConfigTemplate):

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        num_labels: int = 2,
        class_weight: Optional[List[float]] = None,
        multi_label: bool = False,
        analyze_inputs: List[str] = ["a"],
        binary: bool = False,
        node_type: str = None,
        loss_scalar: float = 1.0,
    ):
        super().__init__(
            base="NodeClsTaskHead",
            properties={
                "hidden_size": hidden_size,
                "hidden_dropout_prob": hidden_dropout_prob,
                "num_labels": num_labels,
                "class_weight": class_weight,
                "multi_label": multi_label,
                "analyze_inputs": analyze_inputs,
                "binary": binary,
                "node_type": node_type,
                "loss_scalar": loss_scalar,
            },
        )

    def get_model(self) -> nn.Module:
        if self.properties["multi_label"] == True:
            return MultiLabelNodeClassificationHead(**self.properties)
        else:
            return SingleLabelNodeClassificationHead(**self.properties)


class SiameseGraphClsTaskHeadConfig(ConfigTemplate):

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        num_labels: int = 2,
        class_weight: Optional[List[float]] = None,
        analyze_inputs: List[Tuple[str, str]] = [("a", "b")],
    ):
        super().__init__(
            base="SiameseGraphClsTaskHead",
            properties={
                "hidden_size": hidden_size,
                "hidden_dropout_prob": hidden_dropout_prob,
                "num_labels": num_labels,
                "class_weight": class_weight,
                "analyze_inputs": analyze_inputs,
            },
        )

    def get_model(self) -> nn.Module:
        return SiameseGraphClassificationHead(**self.properties)


HeadConfig = Union[NodeClsTaskHeadConfig, SiameseGraphClsTaskHeadConfig]
