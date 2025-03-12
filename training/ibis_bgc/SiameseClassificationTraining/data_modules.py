import json
import os
from typing import List, Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from omnicons import datdir
from omnicons.collators.MaskCollators import NodeWordMaskCollator
from omnicons.collators.MixedCollators import MixedCollator
from omnicons.collators.NoiseCollators import NodeNoiseCollator
from omnicons.collators.StandardCollators import StandardCollator
from omnicons.specialized_datasets.SiameseClsDataset import (
    ClassificationDynamicDataset,
)


def get_node_vocab(
    vocab_fp: str = os.path.join(
        datdir, "siamese_classification_training", "vocab", "node_vocab.json"
    )
):
    return json.load(open(vocab_fp))


def get_edge_vocab(
    vocab_fp: str = os.path.join(
        datdir, "siamese_classification_training", "vocab", "edge_vocab.json"
    )
):
    return json.load(open(vocab_fp))


class SiameseClsDataModule(LightningDataModule):

    def __init__(
        self,
        cls_dir: str = os.path.join(
            datdir, "siamese_classification_training", "partitions"
        ),
        graph_dir: str = os.path.join(
            datdir, "siamese_classification_training", "graphs"
        ),
        batch_size: int = 10,
        num_workers: int = 0,
        persistent_workers: bool = False,
        subset: Optional[int] = None,
        label_options: List[str] = [
            "label_bin_1",
            "label_bin_2",
            "label_bin_3",
            "label_bin_4",
            "label_bin_5",
        ],
    ):
        super().__init__()
        self.cls_dir = cls_dir
        self.graph_dir = graph_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.subset = subset
        self.label_options = label_options
        # vocab
        self.node_vocab = get_node_vocab()
        self.edge_vocab = get_edge_vocab()
        # collators
        node_types_with_labels = [
            "mibig_chemotype",
            "internal_chemotype",
            "domain",
        ]
        node_types_with_embedding = ["orf", "domain_embedding"]
        self.collator = MixedCollator(
            collators=(
                NodeWordMaskCollator(
                    mask_id=1,
                    p=0.15,
                    node_types_to_consider=node_types_with_labels,
                ),
                NodeNoiseCollator(
                    p=0.15, node_types_to_consider=node_types_with_embedding
                ),
            ),
            standard_collator=StandardCollator(
                variables_to_adjust_by_precision=[
                    ("domain_embedding", "x"),
                    ("orf", "x"),
                ]
            ),
        )

    def get_individual_dataset(self, name: str):
        pairs = pd.read_csv(f"{self.cls_dir}/{name}.csv")
        classification_dict = json.load(
            open(f"{self.cls_dir}/class_dict.json")
        )
        cluster_ids = set(pairs["n1"]) | set(pairs["n2"])
        filename_lookup = {m: f"{self.graph_dir}/{m}.pkl" for m in cluster_ids}
        ds = ClassificationDynamicDataset(
            head_name="bgc_bgc_tanimoto",
            pairs=pairs.to_dict("records"),
            classification_dict=classification_dict,
            root=self.cls_dir,
            label_options=self.label_options,
            subset=self.subset,
            node_vocab=self.node_vocab,
            edge_vocab=self.edge_vocab,
            filename_lookup=filename_lookup,
            in_memory=True,
            dynamic_tensor_render=False,
        )
        return ds

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train = self.get_individual_dataset("train")
            self.val = self.get_individual_dataset("val")
        if stage == "test":
            self.test = self.get_individual_dataset("test")

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
