import json
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from omnicons import datdir, helpers
from omnicons.data_generators.genomic.BiosyntheticWindowDataGenerator import (
    BiosyntheticWindowOrfGraphDataGenerator,
)
from omnicons.data_generators.genomic.utils import (
    chemotypes_to_use,
    subset_genomes,
)


def annotate_data(fn_inputs):
    try:
        (
            rec,
            split,
            meta_df_fp,
            graph_dir,
            subset_dir,
            data_dir,
            ibis_embed_dir,
        ) = fn_inputs
        dg = BiosyntheticWindowOrfGraphDataGenerator(
            meta_df_fp=meta_df_fp,
            graph_dir=graph_dir,
            subset_dir=subset_dir,
            data_dir=data_dir,
            ibis_embed_dir=ibis_embed_dir,
            cluster_chemotypes_to_use=chemotypes_to_use,
        )
        dg.create_class_dicts(
            cluster_chemotype_fp=os.path.join(
                datdir,
                "biosyn_windows",
                "label_files",
                "cluster_chemotype.json",
            )
        )
        dg._prepare_graph_tensor(rec=rec, split=split)
    except KeyError:
        return rec["faa_fh"]
    return None


def create_to_run(
    meta_df_fp: str,
    graph_dir: str,
    subset_dir: str,
    data_dir: str,
    ibis_embed_dir: str,
):
    to_run = []
    for split in ["train", "val", "test"]:
        data_fp = f"{subset_dir}/{split}.csv"
        data = pd.read_csv(data_fp).to_dict("records")
        helpers.create_dir(f"{data_dir}/{split}")
        for rec in data:
            to_run.append(
                [
                    rec,
                    split,
                    meta_df_fp,
                    graph_dir,
                    subset_dir,
                    data_dir,
                    ibis_embed_dir,
                ]
            )
    return to_run


def prepare_split_mapper(graph_dir: str, subset_dir: str, data_dir: str):
    bad = {x: [] for x in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        data_fp = f"{subset_dir}/{split}.csv"
        data = pd.read_csv(data_fp).to_dict("records")
        out = {"quality": {}, "non_quality": {}}
        for rec in tqdm(data):
            faa_basename = os.path.basename(rec["faa_fh"])
            window_quality_lookup_fp = os.path.join(
                graph_dir, faa_basename, "window_info.json"
            )
            try:
                window_quality_lookup = json.load(
                    open(window_quality_lookup_fp, "r")
                )
            except FileNotFoundError:
                bad[split].append(faa_basename)
            for window_idx, dat in window_quality_lookup.items():
                tensor_fp = (
                    f"{data_dir}/{split}/window_{window_idx}_{faa_basename}.pt"
                )
                # skip non-existent tensors
                if not os.path.exists(tensor_fp):
                    continue
                # add chemotype and quality to the appropriate lookup
                chemotypes = dat["window_chemotypes"]
                for chemotype in chemotypes:
                    if out[dat["contig_quality"]].get(chemotype) is None:
                        out[dat["contig_quality"]][chemotype] = []
                    out[dat["contig_quality"]][chemotype].append(tensor_fp)
        # save the lookup to disk
        for category in ["quality", "non_quality"]:
            save_fp = os.path.join(
                subset_dir, f"{split}_contig_{category}.json"
            )
            json.dump(out[category], open(save_fp, "w"))
    save_fp = os.path.join(subset_dir, f"bad_entries.json")
    json.dump(bad, open(save_fp, "w"))


if __name__ == "__main__":
    # The code below was used to generate the splits. it is not
    # necessary to run this code block unless you want to regenerate
    # the splits.
    # subset_genomes(
    #     subset_dir= os.path.join(datdir, "biosyn_windows", "splits"),
    #     df_fp = os.path.join(datdir, "biosyn_windows", "genomes.csv"),
    #     genome_column = "faa_fh",
    #     chemotype_column = "cluster_chemotypes",
    #     split_proportion = 0.10)
    pool = Pool(30)
    # prepare inputs based on splits.
    to_run = create_to_run(
        meta_df_fp=os.path.join(datdir, "biosyn_windows", "genomes.csv"),
        graph_dir=None,  # Replace with the path to the graph directory
        subset_dir=os.path.join(datdir, "biosyn_windows", "splits"),
        data_dir=None,  # replace with the directory to save tensors
        ibis_embed_dir=None,  # Replace with the path to your IBIS outputs
    )
    process = pool.imap_unordered(annotate_data, to_run)
    l = len(to_run)
    [p for p in tqdm(process, total=l)]
    pool.close()
    pool.join()
    # prepares a lookup for the splits to be used in the data module
    # this lookup allows for dynamic loading of the tensors from
    # different datasets on disk. This function produces the  files
    # <split>_contig_quality.json and <split>_contig_non_quality.json
    # provided in the ibis_sm_datasets/biosyn_windows/splits directory
    # via Zenodo.
    prepare_split_mapper(
        graph_dir=None,  # Replace with the path to the graph directory
        subset_dir=os.path.join(datdir, "biosyn_windows", "splits"),
        data_dir=None,  # replace with the directory to save tensors
    )
