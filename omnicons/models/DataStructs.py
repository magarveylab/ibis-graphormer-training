import torch
from dataclasses import dataclass
import itertools as it
from typing import Optional, Dict, List, Tuple

@dataclass
class MultiTaskOutput:
    loss: torch.Tensor
    logits: Dict[str, torch.Tensor]
    labels: Dict[str, torch.Tensor]

    def split(self, tensor_names: List[Tuple[str, str, str]]) -> Tuple[torch.Tensor]:
        out = [self.loss]
        keys = {'logits': self.logits, 'labels': self.labels}
        for hn, inp, k in tensor_names:
            out.append(keys[k].get(f'{hn}___{inp}'))
        return tuple(out)
    
    @classmethod
    def merge(cls, tensors: Tuple[torch.Tensor], tensor_names: List[Tuple[str, str, str]]):
        loss = tensors[0]
        logits = {}
        labels = {}
        for (hn, inp, k), t in zip(tensor_names, tensors[1:]):
            if isinstance(inp, tuple):
                inp = '___'.join(inp)
            if k == 'logits':
                logits[f'{hn}___{inp}'] = t
            else:
                labels[f'{hn}___{inp}'] = t
        return cls(loss=loss, labels=labels, logits=logits)

@dataclass
class ClassificationOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor