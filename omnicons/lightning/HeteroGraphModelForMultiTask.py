from typing import Dict, List, Literal, Optional, Tuple

from omnicons.architectures.HeteroGraphModelForMultiTask import (
    HeteroGraphModelForMultiTask,
)
from omnicons.configs.EdgePoolerConfigs import EdgePoolerConfig
from omnicons.configs.EncoderConfigs import EncoderConfig
from omnicons.configs.GNNConfigs import GNNConfig
from omnicons.configs.GraphPoolerConfigs import GraphPoolerConfig
from omnicons.configs.HeadConfigs import HeadConfig
from omnicons.configs.TransformerConfigs import TransformerConfig
from omnicons.lightning.bases.MultiTaskSingleOptimizerBase import (
    MultiTaskSingleOptimizerBase,
)

NodeType = str
EdgeName = str
EdgeType = Tuple[NodeType, EdgeName, NodeType]
HeadName = str


class HeteroGraphModelForMultiTaskLightning(MultiTaskSingleOptimizerBase):

    def __init__(
        self,
        node_encoders: Dict[NodeType, EncoderConfig] = {},
        edge_encoders: Dict[EdgeName, EncoderConfig] = {},
        edge_type_encoder_config: Optional[EncoderConfig] = None,
        gnn_config: Optional[GNNConfig] = None,
        transformer_config: Optional[TransformerConfig] = None,
        graph_pooler_config: Optional[GraphPoolerConfig] = None,
        edge_pooler_config: Optional[EdgePoolerConfig] = None,
        heads: Dict[HeadName, HeadConfig] = {},
        inputs: List[Literal["a", "b", "c"]] = ["a"],
        edge_types: List[EdgeType] = [],
        **kwargs
    ):
        model = HeteroGraphModelForMultiTask(
            node_encoders=node_encoders,
            edge_encoders=edge_encoders,
            edge_type_encoder_config=edge_type_encoder_config,
            gnn_config=gnn_config,
            transformer_config=transformer_config,
            graph_pooler_config=graph_pooler_config,
            edge_pooler_config=edge_pooler_config,
            heads=heads,
            inputs=inputs,
            edge_types=edge_types,
        )
        super().__init__(model=model, **kwargs)
