import ast
import json
import pickle
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from omnicons import helpers


def bin_number(num, divisor):
    return round(round(num / divisor) * divisor, 1)


def partition_distance_matrix(
    export_dir: str,
    distance_matrix: dict,
    graph_to_chemotype: dict,
    max_subset: int = 10000,
):
    # organized edges by scores
    sorted_edges = {}
    for s1 in tqdm(distance_matrix):
        s1_chemotypes = graph_to_chemotype[s1]
        if (
            "TypeIPolyketide" in s1_chemotypes
            and "NonRibosomalPeptide" in s1_chemotypes
        ):
            chemotype = "Hybrid"
        elif "NonRibosomalPeptide" in s1_chemotypes:
            chemotype = "NonRibosomalPeptide"
        elif "TypeIIPolyketide" in s1_chemotypes:
            chemotype = "TypeIIPolyketide"
        elif "TypeIPolyketide" in s1_chemotypes:
            chemotype = "TypeIPolyketide"
        elif len(s1_chemotypes) > 0:
            chemotype = s1_chemotypes[0]
        else:
            chemotype = "None"
        # sort edges
        for s2, score in distance_matrix[s1].items():
            e1, e2 = sorted([s1, s2])
            rounded_score = bin_number(score, 0.1)
            key = f"{chemotype}_{rounded_score}"
            if key not in sorted_edges:
                sorted_edges[key] = {}
            sorted_edges[key][(e1, e2)] = score
    # split data
    out = {"train": [], "val": [], "test": []}
    for key in tqdm(sorted_edges):
        if len(sorted_edges[key]) < 10:
            continue
        # train test splits
        train_subset, test_subset = train_test_split(
            list(sorted_edges[key].keys()), test_size=0.1
        )
        train_subset, val_subset = train_test_split(
            train_subset, test_size=0.1
        )
        # random shuffles
        random.shuffle(train_subset)
        random.shuffle(val_subset)
        random.shuffle(test_subset)
        # take subsets
        train_subset = list(train_subset)[:max_subset]
        val_subset = list(val_subset)[:max_subset]
        test_subset = list(test_subset)[:max_subset]
        for partition, subset in [
            ("train", train_subset),
            ("val", val_subset),
            ("test", test_subset),
        ]:
            for e1, e2 in subset:
                score = sorted_edges[key][(e1, e2)]
                out[partition].append(
                    {
                        "n1": e1,
                        "n2": e2,
                        "pair_bin": key,
                        "score": score,
                        "label_bin_1": f"score_{bin_number(score, 0.1)}",
                        "label_bin_2": f"score_{bin_number(score, 0.2)}",
                        "label_bin_3": f"score_{bin_number(score, 0.3)}",
                        "label_bin_4": f"score_{bin_number(score, 0.4)}",
                        "label_bin_5": f"score_{bin_number(score, 0.5)}",
                    }
                )
    # create labels
    class_dict = {
        "label_bin_1": set(),
        "label_bin_2": set(),
        "label_bin_3": set(),
        "label_bin_4": set(),
        "label_bin_5": set(),
    }
    for i in out["train"]:
        class_dict["label_bin_1"].add(i["label_bin_1"])
        class_dict["label_bin_2"].add(i["label_bin_2"])
        class_dict["label_bin_3"].add(i["label_bin_3"])
        class_dict["label_bin_4"].add(i["label_bin_4"])
        class_dict["label_bin_5"].add(i["label_bin_5"])
    for x, y in class_dict.items():
        class_dict[x] = {l: idx for idx, l in enumerate(sorted(y))}
    # export partitions
    for partition_name, data in out.items():
        print(f"Partition: {partition_name}, size: {len(data)}")
        pd.DataFrame(data).to_csv(
            f"{export_dir}/{partition_name}.csv", index=None
        )
    # export class dict
    with open(f"{export_dir}/class_dict.json", "w") as json_data:
        json.dump(class_dict, json_data)


cluster_ai_dataset_dir = (
    "/home/gunam/mserv/git_repos/ClusterAI/ClusterAI/dat/datasets"
)


class SiameseClassificationDataGenerator:

    def __init__(
        self,
        export_dir: str = "/home/gunam/storage/data_sets/omnicons/bear_unison",
        kc_fp: str = f"{cluster_ai_dataset_dir}/known_clusters.csv",
        sm_fp: str = f"{cluster_ai_dataset_dir}/smallmolecules.csv",
    ):
        self.export_dir = export_dir
        self.kc_fp = kc_fp
        self.sm_fp = sm_fp
        self.contrastive_dir = f"{self.export_dir}/contrastive"
        # directories
        self.classification_dir = f"{self.export_dir}/classification"
        self.bgc_bgc_tanimoto_dir = (
            f"{self.classification_dir}/bgc_bgc_tanimoto"
        )
        self.bgc_sm_tanimoto_dir = f"{self.classification_dir}/bgc_sm_tanimoto"
        helpers.create_dir(self.bgc_bgc_tanimoto_dir)
        helpers.create_dir(self.bgc_sm_tanimoto_dir)

    def prepare_data(self):
        self.render_chemotypes()
        self.build_bgc_sm_tanimoto_dataset()
        self.build_bgc_bgc_tanimoto_dataset()

    def render_chemotypes(self):
        self.graph_to_chemotype = {}
        for rec in pd.read_csv(self.kc_fp).to_dict("records"):
            cluster_id = "cluster_{}".format(rec["genomic_cluster_id"])
            self.graph_to_chemotype[cluster_id] = ast.literal_eval(
                rec["chemotypes"]
            )

    def build_bgc_sm_tanimoto_dataset(self):
        distance_matrix = pickle.load(
            open(
                f"{self.contrastive_dir}/bgc_sm_tanimoto/distance_matrix.pkl",
                "rb",
            )
        )
        partition_distance_matrix(
            export_dir=self.bgc_sm_tanimoto_dir,
            distance_matrix=distance_matrix,
            graph_to_chemotype=self.graph_to_chemotype,
        )

    def build_bgc_bgc_tanimoto_dataset(self):
        distance_matrix = pickle.load(
            open(
                f"{self.contrastive_dir}/bgc_bgc_tanimoto/distance_matrix.pkl",
                "rb",
            )
        )
        partition_distance_matrix(
            export_dir=self.bgc_bgc_tanimoto_dir,
            distance_matrix=distance_matrix,
            graph_to_chemotype=self.graph_to_chemotype,
        )
