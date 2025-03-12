from typing import Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.poolers.GraphPoolers import NodeClsPooler


class NodeClsPoolerConfig(ConfigTemplate):

    def __init__(self, hidden_channels: int = 128):
        super().__init__(
            base="NodeClsPooler",
            properties={"hidden_channels": hidden_channels},
        )

    def get_model(self) -> nn.Module:
        return NodeClsPooler(**self.properties)


GraphPoolerConfig = Union[NodeClsPoolerConfig,]
