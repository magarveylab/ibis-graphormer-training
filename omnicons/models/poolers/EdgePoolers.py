import torch
from torch import nn

from omnicons.models.encoders.MLPEncoder import MLPEncoder


class EdgeMLPPooler(nn.Module):

    def __init__(
        self,
        node_embed_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.mlp = MLPEncoder(
            input_dim=node_embed_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        x = x.index_select(0, edges.view(-1)).view(
            (edges.shape[0], edges.shape[1], -1)
        )
        x = torch.mean(x, dim=1)
        return self.mlp(x)
