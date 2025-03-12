from typing import Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.transformer.Graphormer import GraphormerBase


class GraphormerConfig(ConfigTemplate):

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        embed_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
    ):
        super().__init__(
            base="Graphormer",
            properties={
                "num_layers": num_layers,
                "num_heads": num_heads,
                "embed_dim": embed_dim,
                "dropout": dropout,
                "attention_dropout": attention_dropout,
                "mlp_dropout": mlp_dropout,
            },
        )

    def get_model(self) -> nn.Module:
        return GraphormerBase(**self.properties)


TransformerConfig = Union[GraphormerConfig]
