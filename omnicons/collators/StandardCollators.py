from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from omnicons.collators.BaseCollators import BaseCollator
from omnicons.data.DataClass import MultiInputData
from torch_geometric.data import Batch, Data, HeteroData

heterogeneous_edge_type = Tuple[str, str, str]


@dataclass
class StandardCollator(BaseCollator):
    variables_to_adjust_by_precision: Union[
        Tuple[str, ...], Tuple[Tuple[str, str], ...]
    ] = ()
    variables_to_adjust_by_index: Union[
        Tuple[str, ...], Tuple[Tuple[heterogeneous_edge_type, str], ...]
    ] = ()
    apply_batch: bool = True

    def postprocess(self, data: MultiInputData) -> Dict[str, torch.Tensor]:
        cache = {}
        for k in data.graphs:
            v = data.graphs[k]
            # homogenous data processing
            if isinstance(v, Data):
                # adjust precision
                for name in self.variables_to_adjust_by_precision:
                    setattr(v, name, getattr(v, name).to(torch.float16))
                # adjust indexes after batching
                _slice_dict = v._slice_dict
                name_list = list((v.keys()))
                for name in name_list:
                    if (
                        "_links" in name
                        or name in self.variables_to_adjust_by_index
                    ):
                        to_correct = getattr(v, name)
                        corrected = self.batch_correction(
                            to_correct=to_correct,
                            _slice_dict=_slice_dict,
                            name=name,
                            is_homogenous=True,
                        )
                        setattr(v, name, corrected)
            # heterogeneous data processing
            elif isinstance(v, HeteroData):
                # adjust precision
                for t, name in self.variables_to_adjust_by_precision:
                    v[t][name] = v[t][name].to(torch.float16)
                # adjust indexes after batching for node features
                _slice_dict = v._slice_dict
                node_types = list(v.node_types)
                for t in node_types:
                    name_list = list(v[t].keys())
                    for name in name_list:
                        if (t, name) in self.variables_to_adjust_by_index:
                            to_correct = v[t][name]
                            corrected = self.batch_correction(
                                to_correct=to_correct,
                                _slice_dict=_slice_dict,
                                name=name,
                                node_type=t,
                                is_homogenous=False,
                            )
                            v[t][name] = corrected
                # adjust indexes after batching for edge features
                edge_types = list(v.edge_types)
                for t in edge_types:
                    name_list = list(v[t].keys())
                    for name in name_list:
                        if (
                            "_links" in name
                            or (t, name) in self.variables_to_adjust_by_index
                        ):
                            to_correct = v[t][name]
                            corrected = self.batch_correction(
                                to_correct=to_correct,
                                _slice_dict=_slice_dict,
                                name=name,
                                edge_type=t,
                                is_homogenous=False,
                            )
                            v[t][name] = corrected
            cache[k] = v
        return MultiInputData(
            graphs=cache, common_y=getattr(data, "common_y", None)
        ).to_dict()

    def batch_correction(
        self,
        to_correct: torch.Tensor,
        _slice_dict: dict,
        name: str,
        edge_type: Optional[heterogeneous_edge_type] = None,
        node_type: Optional[str] = None,
        is_homogenous: bool = True,
    ) -> torch.Tensor:
        # calculate slices from batching
        if is_homogenous:
            slices = _slice_dict[name]
        else:
            slices = _slice_dict[edge_type][name]
        # calculating repeats
        repeats = slices[1:] - slices[:-1]
        dimensions = len(to_correct.shape)
        # single index vector corrections (node siamese tasks)
        if dimensions == 1:
            if is_homogenous:
                adj = torch.repeat_interleave(_slice_dict["x"][:-1], repeats)
            else:
                adj = torch.repeat_interleave(
                    _slice_dict[node_type]["x"][:-1], repeats
                )
        # edge / link correction
        elif dimensions == 2:
            if is_homogenous:
                adj = torch.repeat_interleave(_slice_dict["x"][:-1], repeats)
                adj = adj.repeat(2, 1).transpose(-1, -2)
            else:
                n1, _, n2 = edge_type
                i1_adj = torch.repeat_interleave(
                    _slice_dict[n1]["x"][:-1], repeats
                )
                i2_adj = torch.repeat_interleave(
                    _slice_dict[n2]["x"][:-1], repeats
                )
                adj = torch.cat(
                    [i1_adj.reshape(1, -1), i2_adj.reshape(1, -1)]
                ).transpose(-1, -2)
        # correct indexes to fit the new batched graph
        return torch.add(to_correct, adj)
