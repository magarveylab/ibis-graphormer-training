from typing import Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.poolers.GraphPoolers import (
    HeteroNodeClsPooler,
    NodeClsPooler,
)


class NodeClsPoolerConfig(ConfigTemplate):

    def __init__(self, hidden_channels: int = 128):
        super().__init__(
            base="NodeClsPooler",
            properties={"hidden_channels": hidden_channels},
        )

    def get_model(self) -> nn.Module:
        return NodeClsPooler(**self.properties)


class HeteroNodeClsPoolerConfig(ConfigTemplate):

    def __init__(
        self,
        node_type: str = "",
        index_selector: int = 1,
        hidden_channels: int = 128,
    ):
        super().__init__(
            base="HeteroNodeClsPooler",
            properties={
                "node_type": node_type,
                "index_selector": index_selector,
                "hidden_channels": hidden_channels,
            },
        )

    def get_model(self) -> nn.Module:
        return HeteroNodeClsPooler(**self.properties)


GraphPoolerConfig = Union[NodeClsPoolerConfig, HeteroNodeClsPoolerConfig]
