from typing import Dict, List, Literal, Optional

from omnicons.architectures.GraphModelForMultiTask import GraphModelForMultiTask
from omnicons.configs.EdgePoolerConfigs import EdgePoolerConfig
from omnicons.configs.EncoderConfigs import EncoderConfig
from omnicons.configs.GNNConfigs import GNNConfig
from omnicons.configs.GraphPoolerConfigs import GraphPoolerConfig
from omnicons.configs.HeadConfigs import HeadConfig
from omnicons.configs.TransformerConfigs import TransformerConfig
from omnicons.lightning.bases.MultiTaskSingleOptimizerBase import (
    MultiTaskSingleOptimizerBase,
)

HeadName = str

class GraphModelForMultiTaskLightning(MultiTaskSingleOptimizerBase):

    def __init__(self,
                 node_encoder_config: Optional[EncoderConfig] = None,
                 edge_encoder_config: Optional[EncoderConfig] = None,
                 gnn_config: Optional[GNNConfig] = None,
                 transformer_config: Optional[TransformerConfig] = None,
                 graph_pooler_config: Optional[GraphPoolerConfig] = None,
                 edge_pooler_config: Optional[EdgePoolerConfig] = None,
                 heads: Dict[HeadName, HeadConfig] = {},
                 inputs: List[Literal['a', 'b', 'c']] = ['a'],
                 **kwargs):
        model = GraphModelForMultiTask(node_encoder_config=node_encoder_config,
                                       edge_encoder_config=edge_encoder_config,
                                       gnn_config=gnn_config,
                                       transformer_config=transformer_config,
                                       graph_pooler_config=graph_pooler_config,
                                       edge_pooler_config=edge_pooler_config,
                                       heads=heads,
                                       inputs=inputs)
        super().__init__(model=model, **kwargs)