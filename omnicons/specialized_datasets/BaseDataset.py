from typing import List

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from omnicons.data.DataClass import MultiInputData


class BaseDataset(Dataset):

    def __init__(self, root: str, tensor_ids: List[int]):
        self.root = root
        self.tensor_ids = tensor_ids
        self.tensor_cache = self.get_tensor_cache()
        super().__init__(
            root=root, transform=None, pre_transform=None, pre_filter=None
        )

    def len(self):
        return len(self.tensor_cache)

    @property
    def processed_file_names(self):
        return [f"{self.root}/{tensor_id}.pt" for tensor_id in self.tensor_ids]

    def get_tensor_cache(self):
        return {
            tensor_id: torch.load(f"{self.root}/{tensor_id}.pt")
            for tensor_id in tqdm(self.tensor_ids)
        }

    def get(self, idx: int):
        tensor_id = self.tensor_ids[idx]
        return MultiInputData(graphs={"a": self.tensor_cache[tensor_id]})
