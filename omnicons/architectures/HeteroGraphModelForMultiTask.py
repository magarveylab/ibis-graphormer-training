from typing import Dict, List, Literal, Optional, Tuple, Union

from torch.nn import ModuleDict
from torch_geometric.data import Batch

from omnicons.architectures.forwards.HeteroMultiTaskForward import (
    HeteroMultiTaskForward,
)
from omnicons.configs.EdgePoolerConfigs import EdgePoolerConfig
from omnicons.configs.EncoderConfigs import EncoderConfig
from omnicons.configs.GNNConfigs import GNNConfig
from omnicons.configs.GraphPoolerConfigs import GraphPoolerConfig
from omnicons.configs.HeadConfigs import HeadConfig
from omnicons.configs.TransformerConfigs import TransformerConfig
from omnicons.data.DataClass import (
    batch_to_homogeneous,
    get_lookup_from_hetero,
)
from omnicons.models.encoders.WordEncoder import WordEncoder

NodeType = str
EdgeName = str
EdgeType = Tuple[NodeType, EdgeName, NodeType]
HeadName = str


class HeteroGraphModelForMultiTask(HeteroMultiTaskForward):

    def __init__(
        self,
        node_encoders: Dict[NodeType, EncoderConfig] = None,
        edge_encoders: Dict[EdgeType, EncoderConfig] = None,
        edge_type_encoder_config: EncoderConfig = None,
        gnn_config: Optional[GNNConfig] = None,
        transformer_config: Optional[TransformerConfig] = None,
        graph_pooler_config: Optional[GraphPoolerConfig] = None,
        edge_pooler_config: Optional[EdgePoolerConfig] = None,
        heads: Dict[HeadName, HeadConfig] = {},
        edge_types: List[EdgeType] = [],
        inputs: List[Literal["a", "b", "c"]] = ["a"],
    ):
        super().__init__()
        self.inputs = inputs
        # model architecture
        self.node_encoders = ModuleDict()
        if node_encoders is not None:
            for node_type, node_encoder_config in node_encoders.items():
                self.node_encoders[node_type] = node_encoder_config.get_model()
        self.edge_encoders = ModuleDict()
        if edge_encoders is not None:
            for edge_type, edge_encoder_config in edge_encoders.items():
                self.edge_encoders[edge_type] = edge_encoder_config.get_model()
        self.edge_type_lookup = {e[1]: e for e in edge_types}
        # homo edge encoder is to define edge types
        self.edge_type_encoder = (
            None
            if edge_type_encoder_config == None
            else edge_type_encoder_config.get_model()
        )
        self.gnn = None if gnn_config == None else gnn_config.get_model()
        self.transformer = (
            None
            if transformer_config == None
            else transformer_config.get_model()
        )
        self.graph_pooler = (
            None
            if graph_pooler_config == None
            else graph_pooler_config.get_model()
        )
        self.edge_pooler = (
            None
            if edge_pooler_config == None
            else edge_pooler_config.get_model()
        )
        self.heads = ModuleDict()
        for head_name, head_config in heads.items():
            self.heads[head_name] = head_config.get_model()
        # tensor names
        self.tensor_names = []
        for head_name, head in self.heads.items():
            for inp in head.analyze_inputs:
                if isinstance(inp, tuple):
                    inp = "___".join(inp)
                for key in ["logits", "labels"]:
                    self.tensor_names.append((head_name, inp, key))

    def get_model_outputs(self, data: Batch) -> Batch:
        # clone data as it gets updated via the various steps
        data = data.clone()
        # preprocess node encoding (all types should be converted to same dimensionality)
        for node_type, node_encoder in self.node_encoders.items():
            if isinstance(node_encoder, WordEncoder):
                data[node_type]["x"] = node_encoder(
                    data[node_type]["x"], data[node_type].get("extra_x", None)
                )
            else:  # assumes MLP encoder or SentenceEncoder
                data[node_type]["x"] = node_encoder(data[node_type]["x"])
        # preprocess edge encoding (all types should be converted to same dimensionality)
        # if one edge_type has edge_attr, all edge_types should have encoder
        # otherwise it will be ignored when converting heterogenous data to homogenous
        for edge_name, edge_encoder in self.edge_encoders.items():
            edge_type = self.edge_type_lookup[edge_name]
            if isinstance(edge_encoder, WordEncoder):
                data[edge_type]["edge_attr"] = edge_encoder(
                    data[edge_type]["edge_attr"],
                    data[edge_type].get("extra_edge_attr", None),
                )
            else:  # assumes MLP encoder
                data[edge_type]["edge_attr"] = edge_encoder(
                    data[edge_type]["edge_attr"]
                )
        # convert heterogenous to homogenous
        lookup = get_lookup_from_hetero(data)
        data = batch_to_homogeneous(data)
        # edge encode by edge type
        if self.edge_type_encoder != None:
            data.edge_attr = self.edge_type_encoder(
                data.edge_type, getattr(data, "edge_attr", None)
            )
        # message passing
        if self.gnn != None:
            data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
        # transformer (global attention accross nodes)
        if self.transformer != None:
            data.x = self.transformer(data.x, data.batch)
        # convert homogenous to heterogenous
        data = data.to_heterogeneous()
        # graph readout
        if self.graph_pooler != None:
            if self.graph_pooler.dual == False:
                data.pooled_output = self.graph_pooler(
                    data[lookup[self.graph_pooler.node_type]]["x"],
                    data[lookup[self.graph_pooler.node_type]]["batch"],
                )
            else:
                data.pooled_output = self.graph_pooler(
                    data[lookup[self.graph_pooler.n1_node_type]]["x"],
                    data[lookup[self.graph_pooler.n1_node_type]]["batch"],
                    data[lookup[self.graph_pooler.n2_node_type]]["x"],
                    data[lookup[self.graph_pooler.n2_node_type]]["batch"],
                )
        return (lookup, data)
