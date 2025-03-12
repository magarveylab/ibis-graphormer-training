import os
import pickle
from typing import List

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from omnicons import datdir, helpers
from omnicons.collators.MixedCollators import MixedCollator
from omnicons.collators.NoiseCollators import NodeNoiseCollator
from omnicons.collators.StandardCollators import StandardCollator
from omnicons.data.DatasetWrapper import MultiGraphDataset
from omnicons.specialized_datasets.BaseDataset import BaseDataset


class KFoldDataModule(LightningDataModule):

    def __init__(
        self,
        kfold_iteration: int = 0,
        training_dir: str = os.path.join(datdir, "mibig_training", "training"),
        tensor_dir: str = "/home/gunam/data_sets/mibig_cluster_caller/tensors",
        datasplit_dir: str = os.path.join(
            datdir, "mibig_training", "datasplits"
        ),
        chemotypes_to_consider: List[str] = [
            "NRP",
            "Alkaloid",
            "Polyketide",
            "RiPP",
            "Saccharide",
            "Terpene",
            "Other",
        ],
        suffix_options: List[str] = ["a", "b", "c", "d", "e", "f", "g"],
        batch_size: int = 10,
        num_workers: int = 0,
        persistent_workers: bool = False,
        load_weights: bool = True,
    ):
        super().__init__()
        self.kfold_iteration = kfold_iteration
        self.training_dir = training_dir
        self.tensor_dir = tensor_dir
        self.datasplit_dir = datasplit_dir
        self.chemotypes_to_consider = chemotypes_to_consider
        self.suffix_options = suffix_options
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.load_weights = load_weights
        self.collator = MixedCollator(
            collators=(NodeNoiseCollator(p=0.15),),
            standard_collator=StandardCollator(),
        )
        self.weights_fp = (
            f"{self.training_dir}/{kfold_iteration}/class_weights.pkl"
        )
        helpers.create_dir(f"{self.training_dir}/{kfold_iteration}")

    def get_combined_dataset(self, name: str):
        # build input map
        input_map = {
            c: {"a": s}
            for c, s in zip(self.chemotypes_to_consider, self.suffix_options)
        }
        # build datasets
        ds = {}
        for chemotype in self.chemotypes_to_consider:
            data_fp = f"{self.datasplit_dir}/{chemotype}/{self.kfold_iteration}/{name}.csv"
            tensor_ids = list(pd.read_csv(data_fp).tensor_id)
            ds[chemotype] = BaseDataset(
                root=self.tensor_dir, tensor_ids=tensor_ids
            )
        # combine datasets
        combined_dataset = MultiGraphDataset(datasets=ds, input_map=input_map)
        return combined_dataset

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train = self.get_combined_dataset("train")
            self.val = self.get_combined_dataset(
                "test"
            )  # the val is assumed as test (because its k-fold)
        else:
            self.test = None

    def compute_class_weights(self):
        if (
            os.path.exists(self.weights_fp) == False
            or self.load_weights == False
        ):
            l = self.train.len()
            # collect instances of labels
            labels = {
                chemotype: [] for chemotype in self.chemotypes_to_consider
            }
            for idx in tqdm(range(l), total=l):
                d = self.train.get(idx=idx)
                for chemotype, suffix in zip(
                    self.chemotypes_to_consider, self.suffix_options
                ):
                    labels[chemotype].extend(
                        [i[0] for i in d.graphs[suffix][chemotype].tolist()]
                    )
            # calculate weights
            weights = {}
            for p in labels:
                weights[p] = helpers.get_single_label_class_weight(labels[p])
            with open(self.weights_fp, "wb") as pickle_data:
                pickle.dump(weights, pickle_data)
        else:
            weights = pickle.load(open(self.weights_fp, "rb"))
        return weights

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return train_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return test_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return val_dl
