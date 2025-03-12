import random
from typing import Dict, List, Optional, TypedDict

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from omnicons.data.DataClass import MultiInputData
from omnicons.graphs.HeteroGraph import EdgeVocab, NodeVocab
from omnicons.specialized_datasets.DynamicDataset import DynamicDataset


class ClassificationPair(TypedDict):
    # also contains labels
    n1: int
    n2: int
    pair_bin: str


label_options = [
    "label_bin_1",
    "label_bin_2",
    "label_bin_3",
    "label_bin_4",
    "label_bin_5",
]


class ClassificationDynamicDataset(DynamicDataset):

    def __init__(
        self,
        head_name: str,
        pairs: List[ClassificationPair],
        classification_dict: Dict[str, int],
        root: str,
        label_options: List[str] = label_options,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
        node_vocab: NodeVocab = {},
        edge_vocab: EdgeVocab = {},
        filename_lookup: Dict[str, str] = {},
        in_memory: bool = True,
        dynamic_tensor_render: bool = True,
    ):
        self.root = root
        self.head_name = head_name
        self.label_options = label_options
        self.prepare_data(
            pairs=pairs, classification_dict=classification_dict, subset=subset
        )
        super().__init__(
            root=root,
            node_types_to_consider=node_types_to_consider,
            edge_types_to_consider=edge_types_to_consider,
            node_vocab=node_vocab,
            edge_vocab=edge_vocab,
            filename_lookup=filename_lookup,
            in_memory=in_memory,
            dynamic_tensor_render=dynamic_tensor_render,
        )

    def prepare_data(
        self,
        pairs: List[ClassificationPair],
        classification_dict: Dict[str, int],
        subset: Optional[int],
    ):
        pairs_sorted = {p["pair_bin"]: [] for p in pairs}
        # label lookup
        self.label_lookup = {}
        for p in tqdm(pairs):
            pairs_sorted[p["pair_bin"]].append((p["n1"], p["n2"]))
            if (p["n1"], p["n2"]) not in self.label_lookup:
                self.label_lookup[(p["n1"], p["n2"])] = {}
            for name in self.label_options:
                self.label_lookup[(p["n1"], p["n2"])][name] = (
                    classification_dict[name][p[name]]
                )
        # take subset
        if isinstance(subset, int):
            for b in pairs_sorted:
                pairs_sorted[b] = pairs_sorted[b][:subset]
        self.pairs = [
            {"pair_bin": pair_bin, "pairs": pairs}
            for pair_bin, pairs in pairs_sorted.items()
        ]

    def len(self):
        return len(self.pairs)

    def get(self, idx):
        p1, p2 = random.choice(self.pairs[idx]["pairs"])
        common_y = Data()
        for name in self.label_options:
            label = self.label_lookup[(p1, p2)][name]
            setattr(
                common_y,
                f"{self.head_name}_{name}___a___b",
                torch.LongTensor([[label]]),
            )
        response = MultiInputData(
            graphs={
                "a": self.get_tensor(p1).graphs["a"],
                "b": self.get_tensor(p2).graphs["a"],
            },
            common_y=common_y,
        )
        return response
