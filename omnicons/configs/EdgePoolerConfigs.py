from typing import Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.poolers.EdgePoolers import EdgeMLPPooler


class EdgeMLPPoolerConfig(ConfigTemplate):

    def __init__(
        self,
        node_embed_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__(
            base="EdgeMLPPooler",
            properties={
                "node_embed_dim": node_embed_dim,
                "output_dim": output_dim,
                "dropout": dropout,
                "num_layers": num_layers,
            },
        )

    def get_model(self) -> nn.Module:
        return EdgeMLPPooler(**self.properties)


EdgePoolerConfig = Union[EdgeMLPPoolerConfig]
