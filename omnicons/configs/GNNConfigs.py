from typing import Optional, Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.gnn.GAT import GATBase


class GATConfig(ConfigTemplate):

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        embed_dim: int = 128,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            base="GAT",
            properties={
                "num_layers": num_layers,
                "num_heads": num_heads,
                "embed_dim": embed_dim,
                "edge_dim": edge_dim,
                "dropout": dropout,
            },
        )

    def get_model(self) -> nn.Module:
        return GATBase(**self.properties)


GNNConfig = Union[GATConfig]
