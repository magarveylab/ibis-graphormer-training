import torch
from torch import nn

from omnicons.models.encoders.MLPEncoder import MLPEncoder


class WordEncoder(nn.Module):

    def __init__(
        self,
        num_embeddings: int = 100,
        embedding_dim: int = 128,
        extra_features: int = 0,
        dropout: float = 0.1,
        mlp_layers: int = 1,
    ):
        super().__init__()
        self.encoder = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.extra_features = extra_features
        inp_dim = embedding_dim + extra_features
        self.mlp = MLPEncoder(
            input_dim=inp_dim,
            output_dim=embedding_dim,
            dropout=dropout,
            num_layers=mlp_layers,
        )

    def forward(
        self, x: torch.Tensor, extra_features: torch.Tensor
    ) -> torch.Tensor:
        # process nodes
        x = self.encoder(x.reshape(-1))
        if self.extra_features > 0:
            x = torch.cat((x, extra_features), 1)
        x = self.mlp(x)
        return x
