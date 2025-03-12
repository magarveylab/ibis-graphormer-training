import os
import pickle
from typing import List, TypedDict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from omnicons import datdir
from preprocessing.ibis_sm.mibig_training.utils import get_tensors_from_genome


class OrfInput(TypedDict):
    contig_id: int
    contig_start: int
    contig_stop: int
    embedding: np.array


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def add_embeddings_to_orfs(
    orfs: List[dict], embedding_dir: str  # change this to the appropriate path
):
    embeddings = pickle.load(
        open(os.path.join(embedding_dir, "protein_annotation.pkl"), "rb")
    )
    embedding_lkp = {e["protein_id"]: e["embedding"] for e in embeddings}
    for orf in orfs:
        orf["embedding"] = embedding_lkp.get(orf["protein_id"])
    return orfs


# already provided
def prepare_splits():
    # prepare directories
    summary_fp = f"{datdir}/mibig_training/mibig/summary.csv"
    datasplit_dir = f"{datdir}/mibig_training/datasplits/"
    create_dir(datasplit_dir)
    # partition datasets per chemotype
    data = pd.read_csv(summary_fp).to_dict("records")
    chemotypes_to_eval = [
        "NRP",
        "Alkaloid",
        "Polyketide",
        "RiPP",
        "Saccharide",
        "Terpene",
        "Other",
    ]
    datasets = {}
    for chemotype in tqdm(chemotypes_to_eval):
        create_dir(f"{datasplit_dir}/{chemotype}")
        datasets[chemotype] = []
        for idx, rec in tqdm(enumerate(data), total=len(data), leave=True):
            observed = (
                [] if pd.isna(rec["mibig"]) else rec["mibig"].split("__")
            )
            label = 1 if chemotype in observed else 0
            datasets[chemotype].append(
                {"tensor_id": idx, "filename": rec["filename"], "label": label}
            )
    # stratified kfold splits
    skf = StratifiedKFold(n_splits=5)
    for chemotype, data in datasets.items():
        df = pd.DataFrame(data)
        for i, (train_index, test_index) in enumerate(
            skf.split(df.filename, df.label)
        ):
            create_dir(f"{datasplit_dir}/{chemotype}/{i}")
            train_data = pd.DataFrame([data[idx] for idx in train_index])
            test_data = pd.DataFrame([data[idx] for idx in test_index])
            train_fp = f"{datasplit_dir}/{chemotype}/{i}/train.csv"
            test_fp = f"{datasplit_dir}/{chemotype}/{i}/test.csv"
            if (
                os.path.exists(train_fp) == False
                or os.path.exists(test_fp) == False
            ):
                train_data.to_csv(train_fp, index=None)
                test_data.to_csv(test_fp, index=None)
    # full dataset
    for chemotype, data in datasets.items():
        train, test = train_test_split(data, test_size=0.1, random_state=12)
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
        create_dir(f"{datasplit_dir}/{chemotype}/-1")
        train_fp = f"{datasplit_dir}/{chemotype}/-1/train.csv"
        test_fp = f"{datasplit_dir}/{chemotype}/-1/test.csv"
        if (
            os.path.exists(train_fp) == False
            or os.path.exists(test_fp) == False
        ):
            train_data.to_csv(train_fp, index=None)
            test_data.to_csv(test_fp, index=None)


def prepare_tensors():
    summary_fp = f"{datdir}/mibig_training/summary.csv"
    annotation_dir = f"{datdir}/mibig_training/mibig_csvs"
    # change tensor dir to where you wan tto save them.
    tensor_dir = f"{datdir}/mibig_training/tensors"
    summary_data = pd.read_csv(summary_fp).to_dict("records")
    chemotypes_to_eval = [
        "NRP",
        "Alkaloid",
        "Polyketide",
        "RiPP",
        "Saccharide",
        "Terpene",
        "Other",
    ]
    for tensor_id, rec in tqdm(
        enumerate(summary_data), total=len(summary_data)
    ):
        tensor_fp = f"{tensor_dir}/{tensor_id}.pt"
        if os.path.exists(tensor_fp):
            continue
        fh = rec["filename"]
        orfs = pd.read_csv(f"{annotation_dir}/{fh}.csv").to_dict("records")
        lookup = {
            o["orf_id"]: [] if pd.isna(o["mibig"]) else o["mibig"].split("__")
            for o in orfs
        }
        orfs = add_embeddings_to_orfs(orfs)
        tensor = get_tensors_from_genome(orfs, window_size=400)[0]
        ids = [int(i[0]) for i in tensor.ids]
        for chemotype in chemotypes_to_eval:
            chemotype_feats = torch.LongTensor(
                [[1 if chemotype in lookup[i] else 0] for i in ids]
            )
            setattr(tensor, chemotype, chemotype_feats)
        torch.save(tensor, tensor_fp)


def prepare_datasets():
    prepare_splits()
    prepare_tensors()


if __name__ == "__main__":
    prepare_datasets()
