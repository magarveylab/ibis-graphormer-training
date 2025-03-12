import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
from typing import Optional
from torch_geometric.typing import OptTensor

class GATv2Layer(nn.Module):

    def __init__(self, num_heads: int, embed_dim: int,  edge_dim: Optional[int], dropout: float):
        super().__init__()
        self.conv = GATv2Conv(in_channels=embed_dim,
                              out_channels=embed_dim,
                              heads=num_heads,
                              edge_dim=edge_dim,
                              dropout=dropout,
                              concat=False)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.act = nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: OptTensor) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x

    def jittable(self):
        self.conv = self.conv.jittable('(Tensor, Tensor, OptTensor, OptTensor) -> Tensor')

