import torch
from torch import nn
from torch_geometric.data import Batch

from omnicons.models.transformer.layers.GraphormerLayer import GraphormerLayer

# Code adapted from: https://github.com/rampasek/GraphGPS/blob/main/graphgps/network/graphormer.py


class GraphormerBase(nn.Module):

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        embed_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GraphormerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    mlp_dropout=mlp_dropout,
                )
            )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, batch)
        return x
