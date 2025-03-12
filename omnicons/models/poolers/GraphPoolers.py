import torch
from torch import nn
from torch_geometric.utils import to_dense_batch


class NodeClsPooler(nn.Module):

    def __init__(self, hidden_channels: int = 128, **kwargs):
        super().__init__()
        self.pooler = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # asssumes the first node in DATA is CLS node
        pooled_output = to_dense_batch(x, batch)[0][:, 0]
        return self.pooler(pooled_output)


class HeteroNodeClsPooler(nn.Module):

    def __init__(
        self,
        node_type: str,
        index_selector: int = 1,
        hidden_channels: int = 128,
        **kwargs
    ):
        super().__init__()
        self.node_type = node_type
        self.index_selector = index_selector
        self.pooler = nn.Linear(hidden_channels, hidden_channels)
        self.dual = False

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # asssumes the first node in DATA is CLS node
        pooled_output = to_dense_batch(x, batch)[0][:, self.index_selector]
        return self.pooler(pooled_output)
