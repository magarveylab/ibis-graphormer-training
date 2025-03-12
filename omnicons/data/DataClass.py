import torch
from torch_geometric.data import Data, Batch, HeteroData
from typing import Dict, List, Optional, Union, Tuple

class MultiInputData(Data):

    def __init__(self,
                 graphs: Dict[str, Union[Data, HeteroData]] = {},
                 common_y: Optional[Data] = None):
        super().__init__()
        # example keys for GraphModelForMultiTask: "a", "b", "c" ...
        self.graphs = graphs
        self.common_y = common_y
    
    @property
    def is_hetero(self):
        return True if isinstance(list(self.graphs.values())[0], HeteroData) else False
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        cache = {}
        common_y = getattr(self, 'common_y', None)
        if isinstance(common_y, Data):
            for k, v in common_y.to_dict().items():
                cache[f'common___{k}'] = v
        if isinstance(self.graphs['a'], Data):
            for inp, g in self.graphs.items():
                for k, v in g.to_dict().items():
                    cache[f'batch___{k}___{inp}'] = v
        elif isinstance(self.graphs['a'], HeteroData):
            # its too complicated to break this object
            # since we need to convert this to homogenous and back
            # the recovery function will mess this up
            for inp, g in self.graphs.items():
                cache[inp] = g
        return cache
    
    def get_node_edge_order(self) -> Dict[str, List[Union[str, Tuple[str]]]]:
        '''
        provides the insertion order of nodes and edges added to a HeteroData object, or
        a Data object which was generated from HeteroData.to_homogeneous(). The insertion order
        is critical to maintaining the appropriate tensor indexing corrections for non-x, non-edge type
        tensors in the collator functions.
        '''
        cache = {}
        for key, data in self.graphs.items():
            if isinstance(data, Data):
                cache[key]={
                    'nodes': data._node_type_names,
                    'edges': data._edge_type_names
                }
            elif isinstance(data, HeteroData):
                cache[key] = {
                    'nodes': data.node_types,
                    'edges': data.edge_types
                }
            else:
                raise TypeError('data objects must be torch_geometric.data.HeteroData or torch_geometric.data.Data')
        return cache

def batch_from_data_list(l: List[MultiInputData]) -> MultiInputData:
    keys = l[0].graphs.keys()
    graphs = {k: Batch.from_data_list([d.graphs[k] for d in l]) for k in keys}
    if getattr(l[0], 'common_y', None) == None:
        common_y = None
    else:
        common_y = Batch.from_data_list([d.common_y for d in l])
    return MultiInputData(graphs=graphs, common_y=common_y)

def recover_homogeneous_data(inputs: List[str], **kwargs) -> Dict[str, Data]:
    # parse and organize tensors by input
    batch = {inp: {} for inp in inputs}
    for x, y in kwargs.items():
        prefix = x.split('___')[0]
        suffix = x.split('___')[-1]
        interm = '___'.join(x.split('___')[1:-1])
        if prefix == 'batch' and suffix in batch:
            batch[suffix][interm] = y
    # cast as data objects
    out = {}
    for inp in inputs:
        out[inp] = Data(x=batch[inp]['x'],
                        edge_index=batch[inp]['edge_index'],
                        edge_attr=batch[inp]['edge_attr'])
        for x, y in batch[inp].items():
            if x not in ['x', 'edge_index', 'edge_attr']:
                setattr(out[inp], x, y)
    return out

def get_lookup_from_hetero(h: HeteroData):
    # needed for looking up nodes and edges after converting homogenous to heterogenous
    lookup = {node_type: str(idx) for idx, node_type in enumerate(h.node_types)}
    for idx, edge_type in enumerate(h.edge_types):
        n1 = lookup[edge_type[0]]
        n2 = lookup[edge_type[2]]
        lookup[edge_type] = (n1, str(idx), n2)
    return lookup

def batch_to_homogeneous(batch: Batch, replace_nan_with: Optional[int]= -100, **kwargs) -> Batch:
    data_list = batch.to_data_list()
    data_list = [d.to_homogeneous(**kwargs) for d in data_list]
    batch = Batch.from_data_list(data_list=data_list)
    for k,v in batch.to_dict().items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if torch.is_tensor(v2):
                    batch[k][k2]=torch.nan_to_num(v2, nan = replace_nan_with)
        elif torch.is_tensor(v):
            batch[k]=torch.nan_to_num(v, nan = replace_nan_with)
    return batch