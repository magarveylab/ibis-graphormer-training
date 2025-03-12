import torch
from typing import Tuple

def recast_labels_for_single_label(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # limit input passed to head (skip label -100)
    labels = labels.view(-1)
    consider_index = torch.where(labels != -100, 1, 0).nonzero().view(-1)
    labels_recast = torch.index_select(labels, 0, consider_index)
    return consider_index, labels_recast

def recast_labels_for_multi_label(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # limit input passed to head (skip label -100)
    consider_index = torch.where(labels[:,0] != -100, 1, 0).nonzero().view(-1)
    labels_recast = torch.index_select(labels, 0, consider_index)
    return consider_index, labels_recast.float()

def recast_input_for_single_label(inp: torch.Tensor, consider_index: torch.Tensor) -> torch.Tensor:
    return torch.index_select(inp.view(-1, inp.shape[-1]), 0, consider_index)

def recast_input_for_multi_label(inp: torch.Tensor, consider_index: torch.Tensor) -> torch.Tensor:
    return torch.index_select(inp, 0, consider_index)