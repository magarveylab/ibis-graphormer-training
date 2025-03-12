from typing import Optional

import torch
from torch import nn
from torch_geometric.typing import OptTensor

from omnicons.models.gnn.layers.GATv2Layer import GATv2Layer


class GATBase(nn.Module):

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        embed_dim: int = 128,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Layer(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: OptTensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x

    def jittable(self):
        for layer in self.layers:
            layer.jittable()
