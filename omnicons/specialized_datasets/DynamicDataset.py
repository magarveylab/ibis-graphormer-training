import pickle
from typing import Dict, List, Optional, Union

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from omnicons.graphs.HeteroGraph import EdgeVocab, NodeVocab


class DynamicDataset(Dataset):

    def __init__(
        self,
        root: str,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
        node_vocab: NodeVocab = {},
        edge_vocab: EdgeVocab = {},
        filename_lookup: Dict[str, str] = {},
        in_memory: bool = True,
        dynamic_tensor_render: bool = True,
    ):
        self.node_types_to_consider = node_types_to_consider
        self.edge_types_to_consider = edge_types_to_consider
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.filename_lookup = (
            filename_lookup  # links ids in anchors to graph paths or tensors
        )
        self.in_memory = in_memory
        self.dynamic_tensor_render = dynamic_tensor_render
        if self.in_memory:
            if self.dynamic_tensor_render:
                self.graph_cache = self.get_graph_cache()
                self.tensor_cache = None
            else:
                self.graph_cache = None
                self.tensor_cache = self.get_tensor_cache()
        else:
            self.graph_cache = None
            self.tensor_cache = None
        super().__init__(
            root=root, transform=None, pre_transform=None, pre_filter=None
        )

    @property
    def processed_file_names(self):
        return list(self.filename_lookup.values())

    def get_graph_cache(self):
        cache = {}
        for sample_id, fp in tqdm(self.filename_lookup.items()):
            cache[sample_id] = pickle.load(open(fp, "rb"))
        return cache

    def get_tensor_cache(self):
        cache = {}
        for sample_id, fp in tqdm(self.filename_lookup.items()):
            G = pickle.load(open(fp, "rb"))
            tensor = G.get_tensor_data(
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
            )
            cache[sample_id] = tensor
        return cache

    def get_tensor(self, graph_id: Union[int, str]):
        if self.dynamic_tensor_render:
            # load graph
            if self.in_memory:
                G = self.graph_cache[graph_id]
            else:  # filepath corresponds to HeterGraph
                G = pickle.load(open(self.filename_lookup[graph_id], "rb"))
            # convert to tensor
            return G.get_tensor_data(
                node_vocab=self.node_vocab, edge_vocab=self.edge_vocab
            )
        else:
            # load tensor
            if self.in_memory:
                return self.tensor_cache[graph_id]
            else:
                return torch.load(self.filename_lookup[graph_id])
