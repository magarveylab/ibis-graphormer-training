from omnicons.configs.EncoderConfigs import EncoderConfig
from omnicons.configs.GNNConfigs import GNNConfig
from omnicons.configs.TransformerConfigs import TransformerConfig
from omnicons.configs.GraphPoolerConfigs import GraphPoolerConfig
from omnicons.configs.EdgePoolerConfigs import EdgePoolerConfig
from omnicons.configs.HeadConfigs import HeadConfig
from omnicons.architectures.forwards.MultiTaskForward import MultiTaskForward
from omnicons.models.encoders.WordEncoder import WordEncoder
from torch.nn import ModuleDict
from torch_geometric.data import Batch
from typing import List, Dict, Optional, Literal, Tuple

HeadName = str

class GraphModelForMultiTask(MultiTaskForward):

    def __init__(self,
                 node_encoder_config: Optional[EncoderConfig] = None,
                 edge_encoder_config: Optional[EncoderConfig] = None,
                 gnn_config: Optional[GNNConfig] = None,
                 transformer_config: Optional[TransformerConfig] = None,
                 graph_pooler_config: Optional[GraphPoolerConfig] = None,
                 edge_pooler_config: Optional[EdgePoolerConfig] = None,
                 heads: Dict[HeadName, HeadConfig] = {},
                 inputs: List[Literal['a', 'b', 'c']] = ['a']):
        super().__init__()
        self.inputs = inputs
        # model architecture
        self.node_encoder = None if node_encoder_config == None else node_encoder_config.get_model()
        self.edge_encoder = None if edge_encoder_config == None else edge_encoder_config.get_model()
        self.gnn = None if gnn_config == None else gnn_config.get_model()
        self.transformer = None if transformer_config == None else transformer_config.get_model()
        self.graph_pooler = None if graph_pooler_config == None else graph_pooler_config.get_model()
        self.edge_pooler = None if edge_pooler_config == None else edge_pooler_config.get_model()
        self.heads = ModuleDict()
        for head_name, head_config in heads.items():
            self.heads[head_name] = head_config.get_model()
        # tensor names
        self.tensor_names = []
        for head_name, head in self.heads.items():
            for inp in head.analyze_inputs:
                if isinstance(inp, tuple):
                    inp = '___'.join(inp)
                for key in ['logits', 'labels']:
                    self.tensor_names.append((head_name, inp, key))
                    
    def get_model_outputs(self, data: Batch) -> Batch:
        # clone data as it gets updated via the various steps
        data = data.clone()
        # preprocess node encoding
        if self.node_encoder != None:
            if isinstance(self.node_encoder, WordEncoder):
                data.x = self.node_encoder(data.x, getattr(data, 'extra_x', None))
            else: # assumes MLP encoder
                data.x = self.node_encoder(data.x)
        if self.edge_encoder != None:
            if isinstance(self.edge_encoder, WordEncoder):
                data.edge_attr = self.edge_encoder(data.edge_attr, getattr(data, 'extra_edge_attr', None))
            else: # assumes MLP encoder
                data.edge_attr = self.edge_encoder(data.edge_attr)
        # message passing
        if self.gnn != None:
            data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
        # transformer (global attention accross nodes)
        if self.transformer != None:
            data.x = self.transformer(data.x, data.batch)
        # graph readout
        if self.graph_pooler != None:
            data.pooled_output = self.graph_pooler(data.x, data.batch)
        return data