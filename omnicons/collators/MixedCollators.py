from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from omnicons.collators.BaseCollators import BaseCollator
from omnicons.collators.StandardCollators import StandardCollator
from omnicons.data.DataClass import MultiInputData


@dataclass
class MixedCollator:
    collators: Tuple[BaseCollator, ...] = ()
    standard_collator: StandardCollator = StandardCollator()

    def __call__(
        self, data_list: List[MultiInputData]
    ) -> Dict[str, torch.Tensor]:
        # preprocess
        for d in data_list:
            for k in d.graphs:
                for c in self.collators:
                    d.graphs[k] = c.prepare_individual_data(d.graphs[k])
        return self.standard_collator(data_list)
