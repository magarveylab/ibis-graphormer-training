import math
import os
from glob import glob
from typing import Dict, List, Optional, Union

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm

from omnicons.data.DataClass import MultiInputData

DatasetName = str


class GraphInMemoryDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        filenames: Optional[List[str]] = None,
        subset: Optional[Union[int, float]] = None,
        cache_fp: Optional[str] = None,
    ):
        self.cache_fp = cache_fp
        if isinstance(filenames, list):
            self.filenames = filenames
            if isinstance(subset, int):
                self.filenames = self.filenames[:subset]
            elif isinstance(subset, float):
                n = int(len(self.filenames) * subset)
                self.filenames = self.filenames[:n]
        elif subset is not None:
            if isinstance(subset, int):
                self.filenames = glob(f"{root}/*.pt")[:subset]
            elif isinstance(subset, float):
                filenames = glob(f"{root}/*.pt")
                n = int(len(filenames) * subset)
                self.filenames = filenames[:n]
        else:
            self.filenames = glob(f"{root}/*.pt")
        super().__init__(
            root=root, transform=None, pre_transform=None, pre_filter=None
        )
        if cache_fp == None or os.path.exists(self.cache_fp) == False:
            self.process()
            if cache_fp != None:
                self.save()
        else:
            self.load()

    @property
    def processed_file_names(self):
        return self.filenames

    def process(self):
        # Read data into huge `Data` list.
        data_list = [torch.load(fp) for fp in tqdm(self.filenames)]
        self._data, self.slices = self.collate(data_list)

    def save(self):
        torch.save((self._data, self.slices), self.cache_fp)

    def load(self):
        self._data, self.slices = torch.load(self.cache_fp)


class MultiGraphInMemoryDataset(InMemoryDataset):

    def __init__(
        self,
        datasets: Dict[DatasetName, GraphInMemoryDataset] = {},
        cache_fp: Optional[str] = None,
        input_map: Dict[DatasetName, Dict[str, str]] = {},
    ):
        self.cache_fp = cache_fp
        super().__init__(
            root="", transform=None, pre_transform=None, pre_filter=None
        )
        if self.cache_fp == None or os.path.exists(self.cache_fp) == False:
            self._process(datasets, input_map)
            if cache_fp != None:
                self.save()
        else:
            self.load()

    @property
    def processed_file_names(self):
        return [""]

    def _process(
        self,
        datasets: Dict[DatasetName, GraphInMemoryDataset],
        input_map: Dict[DatasetName, Dict[str, str]],
    ):
        # find largest dataset
        max_n = max([d.len() for d in datasets.values()])
        # dataset to indexes
        dataset_indexes = {
            name: list(range(d.len())) for name, d in datasets.items()
        }
        for name, indexes in dataset_indexes.items():
            repeats = math.floor(max_n / len(indexes))
            remainder = max_n % len(indexes)
            dataset_indexes[name] = indexes * repeats + indexes[:remainder]
        # create new data objects
        data_list = []
        keys = list(dataset_indexes.keys())
        values = list(dataset_indexes.values())
        zipped = list(zip(*values))
        for dp in zipped:
            graphs = {}
            common_y = Data()
            for name, idx in zip(keys, dp):
                d = datasets[name].get(idx)
                # add to graphs
                for old_inp, s in d.graphs.items():
                    if old_inp not in input_map[name]:
                        continue
                    new_inp = input_map[name][old_inp]
                    graphs[new_inp] = s
                # add to common y
                if hasattr(d, "common_y"):
                    for k, v in d.common_y.to_dict().items():
                        old_inps = k.split("___")[1:]
                        new_inps = [
                            input_map[name][inp]
                            for inp in old_inps
                            if inp in input_map[name]
                        ]
                        if len(new_inps) != len(old_inps):
                            continue
                        new_k = (
                            k.split("___")[0] + "___" + "___".join(new_inps)
                        )
                        setattr(common_y, new_k, v)
            data_list.append(MultiInputData(graphs=graphs, common_y=common_y))
        # collate
        self._data, self.slices = self.collate(data_list)

    def save(self):
        torch.save((self._data, self.slices), self.cache_fp)

    def load(self):
        self._data, self.slices = torch.load(self.cache_fp)


class GraphDataset(Dataset):

    def __init__(
        self,
        root: str,
        filenames: Optional[List[str]] = None,
        subset: Optional[Union[int, float]] = None,
    ):
        if isinstance(filenames, list):
            self.filenames = filenames
            if isinstance(subset, int):
                self.filenames = self.filenames[:subset]
            elif isinstance(subset, float):
                n = int(len(self.filenames) * subset)
                self.filenames = self.filenames[:n]
        elif subset is not None:
            if isinstance(subset, int):
                self.filenames = glob(f"{root}/*.pt")[:subset]
            elif isinstance(subset, float):
                filenames = glob(f"{root}/*.pt")
                n = int(len(filenames) * subset)
                self.filenames = filenames[:n]
        else:
            self.filenames = sorted(glob(f"{root}/*.pt"))
        super().__init__(
            root=root, transform=None, pre_transform=None, pre_filter=None
        )
        self.lookup = {idx: fp for idx, fp in enumerate(self.filenames)}

    @property
    def processed_file_names(self):
        return self.filenames

    def len(self):
        return len(self.filenames)

    def get(self, idx):
        return torch.load(self.lookup[idx])


class MultiGraphDataset(Dataset):

    def __init__(
        self,
        datasets: Dict[DatasetName, GraphDataset] = {},
        cache_fp: Optional[str] = None,
        input_map: Dict[DatasetName, Dict[str, str]] = {},
    ):
        self.cache_fp = cache_fp
        super().__init__(
            root="", transform=None, pre_transform=None, pre_filter=None
        )
        self.input_map = input_map
        self._process(datasets=datasets)
        self.datasets = datasets

    @property
    def processed_file_names(self):
        return [""]

    def len(self):
        return len(self.zipped)

    def _process(self, datasets: Dict[DatasetName, GraphDataset]):
        # find largest dataset
        max_n = max([d.len() for d in datasets.values()])
        # dataset to indexes
        dataset_indexes = {
            name: list(range(d.len())) for name, d in datasets.items()
        }
        for name, indexes in dataset_indexes.items():
            repeats = math.floor(max_n / len(indexes))
            remainder = max_n % len(indexes)
            dataset_indexes[name] = indexes * repeats + indexes[:remainder]
        # keep keys
        self.keys = list(dataset_indexes.keys())
        values = list(dataset_indexes.values())
        # zip stepped indices
        self.zipped = list(zip(*values))

    def get(self, idx):
        dp = self.zipped[idx]
        graphs = {}
        common_y = Data()
        for name, dp_idx in zip(self.keys, dp):
            d = self.datasets[name].get(idx=dp_idx)
            # add to graphs
            for old_inp, s in d.graphs.items():
                if old_inp not in self.input_map[name]:
                    continue
                new_inp = self.input_map[name][old_inp]
                graphs[new_inp] = s
            # add to common y
            if hasattr(d, "common_y"):
                for k, v in d.common_y.to_dict().items():
                    old_inps = k.split("___")[1:]
                    new_inps = [
                        self.input_map[name][inp]
                        for inp in old_inps
                        if inp in self.input_map[name]
                    ]
                    if len(new_inps) != len(old_inps):
                        continue
                    new_k = k.split("___")[0] + "___" + "___".join(new_inps)
                    setattr(common_y, new_k, v)
        return MultiInputData(graphs=graphs, common_y=common_y)
