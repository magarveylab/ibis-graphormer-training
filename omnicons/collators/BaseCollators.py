from omnicons.data.DataClass import MultiInputData, batch_from_data_list
from torch_geometric.data import Data, HeteroData
from dataclasses import dataclass
import torch
from typing import List, Union, Dict

@dataclass
class BaseCollator:
    apply_batch: bool = False
    
    def __call__(self, data_list: List[MultiInputData]) -> Union[Dict[str, torch.Tensor], List[MultiInputData]]:
        # preprocess
        for d in data_list:
            for k in d.graphs:
                d.graphs[k] = self.prepare_individual_data(d.graphs[k])
        # batch data
        if self.apply_batch:
            out = batch_from_data_list(data_list)
            # postprocess
            return self.postprocess(out)
        else:
            return data_list
            
    def prepare_individual_data(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        # method is overwritten (includes self-supervised training tasks)
        return data
    
    def postprocess(self, data: MultiInputData) -> Dict[str, torch.Tensor]:
        # method is overwritten (post index corrections after batching)
        return data.to_dict()