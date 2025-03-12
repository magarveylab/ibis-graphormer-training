import torch
from torch import nn

from omnicons.models.encoders.layers.LinearLayer import LinearLayer


class MLPEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            LinearLayer(
                in_channels=input_dim, out_channels=output_dim, dropout=dropout
            )
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                LinearLayer(
                    in_channels=output_dim,
                    out_channels=output_dim,
                    dropout=dropout,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
