from typing import Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.encoders.MLPEncoder import MLPEncoder
from omnicons.models.encoders.WordEncoder import WordEncoder


class MLPEncoderConfig(ConfigTemplate):

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__(
            base="MLPEncoder",
            properties={
                "input_dim": input_dim,
                "output_dim": output_dim,
                "dropout": dropout,
                "num_layers": num_layers,
            },
        )

    def get_model(self) -> nn.Module:
        return MLPEncoder(**self.properties)


class WordEncoderConfig(ConfigTemplate):

    def __init__(
        self,
        num_embeddings: int = 100,
        embedding_dim: int = 128,
        extra_features: int = 0,
        dropout: float = 0.1,
        mlp_layers: int = 1,
    ):
        super().__init__(
            base="WordEncoder",
            properties={
                "num_embeddings": num_embeddings,
                "embedding_dim": embedding_dim,
                "extra_features": extra_features,
                "dropout": dropout,
                "mlp_layers": mlp_layers,
            },
        )

    def get_model(self) -> nn.Module:
        return WordEncoder(**self.properties)


EncoderConfig = Union[MLPEncoderConfig, WordEncoderConfig]
