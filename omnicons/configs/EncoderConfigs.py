from typing import Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.encoders.MLPEncoder import MLPEncoder


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


EncoderConfig = Union[MLPEncoderConfig]
