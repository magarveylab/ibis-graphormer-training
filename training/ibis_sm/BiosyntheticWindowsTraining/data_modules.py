import json
import os
import pickle
from typing import List, Literal, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from omnicons import datdir, helpers
from omnicons.collators.MixedCollators import MixedCollator
from omnicons.collators.MultiLabelCollators import MultiLabelCollator
from omnicons.collators.NoiseCollators import NodeNoiseCollator
from omnicons.collators.StandardCollators import StandardCollator
from omnicons.data.DatasetWrapper import GraphDataset, MultiGraphDataset

datasets_to_consider = ["quality_biosyn_window", "non_quality_biosyn_window"]

suffix_options = ["a", "b"]


class BiosyntheticWindowMultiDataModule(LightningDataModule):
    def __init__(
        self,
        cache_dir: str,  # path to tensor cache.
        labels_to_consider: Tuple[
            Literal["boundary_cat", "orf_chemotype", "clust_chemotype"]
        ],
        datasets_to_consider: List[str] = datasets_to_consider,
        subset: Optional[int] = None,
        batch_size: int = 40,
        num_workers: int = 0,
        persistent_workers: bool = False,
        weights_fp: str = os.path.join(
            datdir, "biosyn_windows", "biosyn_windows_weights.pkl"
        ),
        load_weights: bool = True,
        export_weights: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.collator = MixedCollator(
            collators=(
                MultiLabelCollator(
                    multilabel_properties={
                        "boundary_cat": "node",
                        "clust_chemotype": "node",
                    },
                    labels_to_sparsify=(),
                    labels_to_consider=labels_to_consider,
                ),
                NodeNoiseCollator(p=0.15),
            ),
            standard_collator=StandardCollator(),
        )
        self.weights_fp = weights_fp
        self.load_weights = load_weights
        self.export_weights = export_weights
        self.datasets_to_consider = datasets_to_consider
        self.subset = subset
        self.cache_dir = cache_dir

    def get_individual_datasets(self, split_name: str):
        # get quality and non-quality dynamic datasets here.
        ds = {}
        if "quality_biosyn_window" in self.datasets_to_consider:
            fn_lookup_fp = os.path.join(
                datdir,
                "biosyn_windows",
                "splits" f"{split_name}_contig_quality.json",
            )
            fn_lookup = json.load(open(fn_lookup_fp, "r"))
            filenames = set()
            tmp = [filenames.update(x) for x in fn_lookup.values()]
            filenames = list(filenames)
            ds["quality_biosyn_window"] = GraphDataset(
                root=None,
                filenames=filenames,
                subset=self.subset,
            )
        if "non_quality_biosyn_window" in self.datasets_to_consider:
            fn_lookup_fp = os.path.join(
                datdir,
                "biosyn_windows",
                "splits" f"{split_name}_contig_non_quality.json",
            )
            fn_lookup = json.load(open(fn_lookup_fp, "r"))
            filenames = set()
            tmp = [filenames.update(x) for x in fn_lookup.values()]
            filenames = list(filenames)
            ds["non_quality_biosyn_window"] = GraphDataset(
                root=None,
                filenames=filenames,
                subset=self.subset,
            )
        return ds

    def set_dataset(self, split_name: str):
        self.input_map = {
            ds_name: {"a": s}
            for ds_name, s in zip(self.datasets_to_consider, suffix_options)
        }
        ds = self.get_individual_datasets(split_name)
        dataset = MultiGraphDataset(datasets=ds, input_map=self.input_map)
        del ds
        setattr(self, split_name, dataset)

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.set_dataset("train")
            self.set_dataset("val")
        if stage == "test":
            self.set_dataset("test")

    def compute_class_weights(self):
        if (
            os.path.exists(self.weights_fp) == False
            or self.load_weights == False
        ):
            clust_ctype_labels = helpers.NodeSingleLabelDict(ignore_index=-100)
            boundary_cat_labels = helpers.NodeSingleLabelDict(
                ignore_index=-100
            )
            l = self.train.len()
            for idx in tqdm(range(l), total=l):
                d = self.train.get(idx=idx)
                clust_ctype_labels.update(d.graphs["a"].clust_chemotype)
                boundary_cat_labels.update(d.graphs["a"].boundary_cat)
            weights_dict = {
                # 'orf_chemotype': orf_ctype_labels,
                "cluster_chemotype": clust_ctype_labels,
                "boundary_cat": boundary_cat_labels,
            }
            pickle.dump(
                weights_dict,
                open(
                    os.path.join(
                        os.path.dirname(self.weights_fp), "weights_objects.pkl"
                    ),
                    "wb",
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            weights = {
                "cluster_chemotype": clust_ctype_labels.get_class_weight(),
                # only keep the positive value weight, since it's a single class binary label.
                "boundary_cat": [boundary_cat_labels.get_class_weight()[1]],
            }
            if self.export_weights == True:
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
