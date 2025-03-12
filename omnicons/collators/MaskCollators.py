import math
from dataclasses import dataclass
from typing import Dict, Union

import torch
from torch_geometric.data import Data, HeteroData

from omnicons.collators.BaseCollators import BaseCollator


@dataclass
class NodeWordMaskCollator(BaseCollator):
    # in word graphs, each node is represented by a word
    # random subset of words are replaced with masked nodes
    # mask_name should correspond to a SingleNodelLabelClassificationHead
    mask_id: int = 1
    p: int = 0.15
    mask_name: str = "node_mask"
    apply_batch: bool = False
    node_types_to_consider: tuple = ()

    def prepare_individual_data(
        self, data: Union[Data, HeteroData]
    ) -> Union[Data, HeteroData]:
        data = data.clone()
        if isinstance(data, Data):
            out = self.process(data.x)
            for k, v in out.items():
                setattr(data, k, v)
        elif isinstance(data, HeteroData):
            for node_type in data.node_types:
                if node_type not in self.node_types_to_consider:
                    continue
                out = self.process(data[node_type].x)
                for k, v in out.items():
                    setattr(data[node_type], k, v)
        return data

    def process(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        l = x.shape[0]
        # generate mask
        masked_nodes = math.ceil(l * self.p)
        mask = torch.cat(
            [
                torch.ones(masked_nodes, dtype=torch.bool),
                torch.zeros(l - masked_nodes, dtype=torch.bool),
            ]
        )
        mask = mask.index_select(0, torch.randperm(mask.shape[0]))
        # introduce mask nodes
        y = x.reshape(-1).clone()
        x[mask] = torch.tensor([self.mask_id])
        y[~mask] = -100  # non-masked words are ignored
        return {"x": x, self.mask_name: y}
