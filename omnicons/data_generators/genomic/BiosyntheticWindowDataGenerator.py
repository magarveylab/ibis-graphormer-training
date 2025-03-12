import json
import os
import pickle
from glob import glob
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
import xxhash
from tqdm import tqdm

from omnicons import datdir, helpers
from omnicons.data.DataClass import MultiInputData
from omnicons.data_generators.genomic.utils import (
    get_prism_cluster_orfs,
    subset_genomes,
)
from omnicons.graph_converters.homogeneous import genome_graph


class BiosyntheticWindowOrfGraphDataGenerator:
    def __init__(
        self,
        meta_df_fp: str = os.path.join(
            datdir, "biosyn_windows", "genomes.csv"
        ),
        graph_dir: str = None,  # Replace with the path to the graph directory
        subset_dir: str = os.path.join(datdir, "biosyn_windows", "splits"),
        ibis_embed_dir: str = None,  # Replace with the path to your IBIS outputs
        data_dir: str = None,  # replace with the directory to save tensors
        cluster_chemotypes_to_use: List[int] = None,
        embedding_dim: int = 1024,
        max_size: int = 200,
        wrap_multi_input: bool = True,
        ignore_index: int = -100,
    ):
        self.meta_df_fp = meta_df_fp
        self.subset_dir = subset_dir
        self.ibis_embed_dir = ibis_embed_dir
        self.data_dir = data_dir
        self.graph_dir = graph_dir
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.class_dicts = {}
        self.wrap_multi_input = wrap_multi_input
        self.ignore_index = ignore_index
        self.cluster_chemotypes_to_use = cluster_chemotypes_to_use

    def prepare_data(self):
        self.create_class_dicts()
        subset_genomes(
            subset_dir=self.subset_dir,
            df_fp=self.meta_df_fp,
            genome_column="faa_fh",  # will split by this column
            chemotype_column="cluster_chemotypes",
            split_proportion=0.1,
        )
        self.prepare_graph_tensors()

    def create_class_dicts(
        self,
        cluster_chemotype_fp=os.path.join(
            datdir,
            "biosyn_windows",
            "label_files",
            "cluster_chemotype.json",
        ),
    ):
        if (
            os.path.isfile(cluster_chemotype_fp)
            and cluster_chemotype_fp is not None
        ):
            self.class_dicts["cluster_chemotype"] = json.load(
                open(cluster_chemotype_fp, "r")
            )
        else:
            self.create_cluster_chemotype_class_labels(
                save_fp=cluster_chemotype_fp
            )

    def create_cluster_chemotype_class_labels(
        self,
        save_fp: str = None,
        overwrite: bool = False,
        add_hybrid: bool = True,
    ):
        save_fp = (
            save_fp
            if save_fp is not None
            else os.path.join(
                os.path.dirname(self.data_dir), "chemotype_labels.json"
            )
        )
        if os.path.isfile(save_fp) and not overwrite:
            self.class_dicts["cluster_chemotype"] = json.load(
                open(save_fp, "r")
            )
        else:
            from omnicons.data_generators.genomic.utils import (
                universal_chemotypes_w_hybrid,
            )

            all_ctypes = universal_chemotypes_w_hybrid
            if not add_hybrid:
                all_ctypes = list(set(all_ctypes) - set(["PKSNRPSHybrid"]))
            if self.cluster_chemotypes_to_use is not None:
                ctypes_to_consider = set(all_ctypes).intersection(
                    self.cluster_chemotypes_to_use
                )
                assert len(ctypes_to_consider) == len(
                    self.cluster_chemotypes_to_use
                )
            else:
                ctypes_to_consider = all_ctypes
            self.class_dicts["cluster_chemotype"] = {
                x: i for i, x in enumerate(sorted(ctypes_to_consider))
            }
        if not os.path.isfile(save_fp) or overwrite:
            json.dump(
                self.class_dicts["cluster_chemotype"], open(save_fp, "w")
            )

    def prepare_graph_tensors(
        self, splits: List[str] = ["train", "val", "test"]
    ):
        for split in tqdm(splits):
            data_fp = f"{self.subset_dir}/{split}.csv"
            data = pd.read_csv(data_fp).to_dict("records")
            helpers.create_dir(f"{self.data_dir}/{split}")
            for rec in tqdm(data, total=len(data), leave=True):
                self._prepare_graph_tensor(split=split, rec=rec)

    def _prepare_graph_tensor(
        self, rec: Dict, split: Literal["train", "test", "val"]
    ):
        faa_basename = os.path.basename(rec["faa_fh"])
        window_fps = glob(
            os.path.join(self.graph_dir, faa_basename, "*.nx.pkl")
        )
        # Embeddings have not been assigned, so create lookup
        embed_fp = os.path.join(
            self.ibis_embed_dir, faa_basename, "protein_annotation.pkl"
        )
        if not os.path.exists(embed_fp):
            return None
        self.embed_lookup = {
            xxhash.xxh32(
                annot["sequence"].replace("*", "")
            ).intdigest(): annot["embedding"]
            for annot in pickle.load(open(embed_fp, "rb"))
        }
        for window_fp in window_fps:
            window_idx = int(
                os.path.basename(window_fp.replace(".nx.pkl", ""))
            )
            export_fp = f"{self.data_dir}/{split}/window_{window_idx}_{faa_basename}.pt"
            if os.path.exists(export_fp):
                return None
            try:
                self.G = pickle.load(open(window_fp, "rb"))
                if self.G.number_of_nodes() == 0:
                    return None
            except FileNotFoundError:
                return None
            # assign embeddings
            for n in self.G.nodes():
                peptide_id = self.G.nodes[n]["protein_id"]
                embed = torch.tensor(self.embed_lookup[peptide_id])
                self.G.nodes[n]["embedding"] = embed
            nodes = sorted(self.G.nodes(), key=lambda n: n)
            self.node_to_idx = {n: i for i, n in enumerate(nodes)}
            n_clust_chemotype_labels, boundary_cats = self.assign_node_labels(
                nodes=nodes
            )
            x = torch.stack(
                [self.G.nodes[n]["embedding"] for n in nodes], dim=0
            )
            dat = genome_graph.create_tensor_from_graph(
                graph=self.G, x=x, node_to_idx=self.node_to_idx
            )
            dat.clust_chemotype = n_clust_chemotype_labels
            dat.boundary_cat = boundary_cats
            if self.wrap_multi_input:
                dat = MultiInputData(graphs={"a": dat})
            torch.save(dat, export_fp)

    def assign_node_labels(self, nodes):
        ## create single label empty arrays. ##
        boundary_cats = np.zeros((len(nodes)))
        n_cluster_chemotype_labels = np.zeros(len(nodes))
        ## fill arrays with true label values ##
        for n in nodes:
            node_props = self.G.nodes[n]
            node_index = self.node_to_idx[n]
            ## Assign multi-class single labels ##
            cluster_chemotype = node_props.get(
                "cluster_chemotype", self.ignore_index
            )
            n_cluster_chemotype_labels[node_index] = self.class_dicts[
                "cluster_chemotype"
            ].get(cluster_chemotype, self.ignore_index)
            ## add binary labels ##
            boundary_category = node_props.get("cluster_node_cat")
            if boundary_category == "core":
                boundary_cats[node_index] = 1
            elif boundary_category == "peripheral":
                boundary_cats[node_index] = 0
            else:
                boundary_cats[node_index] = self.ignore_index
        return (
            torch.LongTensor(n_cluster_chemotype_labels),
            torch.LongTensor(boundary_cats),
        )
