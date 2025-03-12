import math
from dataclasses import dataclass
from typing import Dict, Union

import torch
from torch_geometric.data import Data, HeteroData

from omnicons.collators.BaseCollators import BaseCollator


@dataclass
class NodeNoiseCollator(BaseCollator):
    # in graphs with pre-loaded embeddings, noise is added to those embeddings
    # and recovered based on context (self-supervised learning tasks)
    p: int = 0.15
    noise_name: str = "noise"
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
        dim = x.shape[1]
        masked_nodes = math.ceil(l * self.p)
        # generate mask
        mask = torch.cat(
            [
                torch.ones(masked_nodes, dtype=torch.bool),
                torch.zeros(l - masked_nodes, dtype=torch.bool),
            ]
        )
        mask = mask.index_select(0, torch.randperm(mask.shape[0]))
        # introduce noise
        y = x.clone()
        x[mask] = x[mask] + (0.1**0.5) * torch.randn(dim).reshape(1, -1)
        return {
            "x": x,
            self.noise_name: y,
            f"{self.noise_name}_target": mask.unsqueeze(dim=1),
        }
