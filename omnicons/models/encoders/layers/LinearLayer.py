import torch
import torch.nn.functional as F
from torch import nn


class LinearLayer(nn.Module):

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
