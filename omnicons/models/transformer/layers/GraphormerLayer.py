from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

# code adapted from https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/graphormer_layer.py


class GraphormerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        mlp_dropout: float,
    ):
        """Implementation of the Graphormer layer.
        This layer is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.
        Args:
            embed_dim: The number of hidden dimensions of the model
            num_heads: The number of heads of the Graphormer model
            dropout: Dropout applied after the attention and after the MLP
            attention_dropout: Dropout applied within the attention
            input_dropout: Dropout applied within the MLP
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, attention_dropout, batch_first=True
        )
        self.input_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # We follow the paper in that all hidden dims are
        # equal to the embedding dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, initial_x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_norm(initial_x)
        x, real_nodes = to_dense_batch(x, batch)
        x = self.attention(x, x, x, ~real_nodes)[0][real_nodes]
        x = self.dropout(x) + initial_x
        x = self.mlp(x) + x
        return x
