import json
import os
import pickle
from typing import Dict, List, Literal, Optional, Set, TypedDict, Union

import more_itertools as mit
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from omnicons.data_generators.genomic.utils import (
    DomainData,
    OrfData,
    get_antismash_clusters,
    get_prism_cluster_orfs,
)


class OrfGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.contig_to_nodes = {}

    @property
    def core_orfs(self):
        return set(
            n
            for n in self.G.nodes
            if self.G.nodes[n]["cluster_node_cat"] == "core"
        )

    def add_orfs(
        self,
        orfs: List[OrfData],
        tolerance=10000,
        diff_method: Literal["raw", "unannotated"] = "raw",
        force_perfect_domain_edges: bool = True,
        include_domains: bool = True,
    ):
        self.diff_method = diff_method
        if force_perfect_domain_edges:
            self.orf_to_domains = {}
        orfs = sorted(orfs, key=lambda x: (x["contig_id"], x["start"]))
        # print('Adding nodes...')
        node_id = 1
        for orf in tqdm(orfs):
            contig_id = orf["contig_id"]
            orf_start = orf["start"]
            orf_stop = orf["stop"]
            if orf["domains"] == [] or include_domains is False:
                self.G.add_node(
                    node_id,
                    contig_id=contig_id,
                    orf_id=orf["orf_id"],
                    protein_id=orf["protein_id"],
                    start=orf_start,
                    stop=orf_stop,
                    embedding=orf["embedding"],
                    ec_number=orf["ec_number"],
                    ec_homology_score=orf["ec_homology_score"],
                    node_type="orf",
                    cluster_node_cat=None,
                )
                if contig_id not in self.contig_to_nodes:
                    self.contig_to_nodes[contig_id] = []
                self.contig_to_nodes[contig_id].append(node_id)
                node_id += 1
            else:
                # NOTE: this will not add the orf containing these domains
                for domain in orf["domains"]:
                    domain_contig_id = domain["contig_id"]
                    domain_orf_id = domain["orf_id"]
                    self.G.add_node(
                        node_id,
                        contig_id=domain_contig_id,
                        orf_id=domain_orf_id,
                        protein_id=domain["protein_id"],
                        # domains indexed from start of orf, convert to
                        # global index
                        start=orf_start + domain["start"],
                        stop=orf_start + domain["stop"],
                        embedding=domain["embedding"],
                        node_type="domain",
                        cluster_node_cat=None,
                    )
                    if domain_contig_id not in self.contig_to_nodes:
                        self.contig_to_nodes[domain_contig_id] = []
                    self.contig_to_nodes[domain_contig_id].append(node_id)
                    if domain_orf_id not in self.orf_to_domains:
                        self.orf_to_domains[domain_orf_id] = {
                            "orf_start": orf_start,
                            "orf_stop": orf_stop,
                            "domain_nodes": [],
                        }
                    self.orf_to_domains[domain_orf_id]["domain_nodes"].append(
                        node_id
                    )
                    node_id += 1
        for contig_id, node_ids in self.contig_to_nodes.items():
            # print(f'Adding edges for contig {contig_id}')
            self.add_weighted_edges(
                node_ids=node_ids, tolerance=tolerance, diff_method=diff_method
            )
        if force_perfect_domain_edges:
            # sort nodes by node ID
            for orf_id, domain_dict in self.orf_to_domains.items():
                domain_nodes = domain_dict["domain_nodes"]
                for n1 in domain_nodes:
                    for n2 in domain_nodes[1:]:
                        self.G.add_edge(n1, n2, score=1)
        return None

    def add_weighted_edges(
        self,
        node_ids: List[int],
        tolerance: Optional[int] = None,
        diff_method: Optional[Literal["raw", "unannotated"]] = None,
    ):
        tolerance = self.tolerance if tolerance is None else tolerance
        diff_method = self.diff_method if diff_method is None else diff_method
        node_ids = sorted(node_ids, key=lambda x: self.G.nodes[x]["start"])
        for idx, o1 in enumerate(node_ids):
            for o2 in node_ids[idx + 1 :]:
                diff = self.get_diff(o1, o2, method=diff_method)
                if diff <= tolerance:
                    score = round((tolerance - diff) / tolerance, 2)
                    self.G.add_edge(o1, o2, score=score)
                # since orfs are sequential, once the difference is
                # exceeded, it will be for all subsequent orfs as well.
                else:
                    break

    def get_diff(self, n1, n2, method: Literal["raw", "unannotated"]):
        default_diff = self.G.nodes[n2]["start"] - self.G.nodes[n1]["stop"]
        default_diff = default_diff if default_diff >= 0 else 0
        if method == "raw":
            diff = default_diff
        elif method == "unannotated":
            if n2 - n1 == 1:
                diff = default_diff
            else:
                # get the total length of annotated orfs between the two
                #  nodes of interest
                part_annot = []
                n1_stop = self.G.nodes[n1]["stop"]
                n2_start = self.G.nodes[n2]["start"]
                for n in range(n1 + 1, n2, 1):
                    nxt_start = self.G.nodes[n]["start"]
                    prev_stop = self.G.nodes[n]["stop"]
                    # sometimes annots overlap. For score purposes,
                    # force them to not overlap. Will not affect raw
                    # annots.
                    if nxt_start > n2_start:
                        nxt_start = n2_start
                    if prev_stop < n1_stop:
                        prev_stop = n1_stop
                    part_annot.append(nxt_start - prev_stop)
                annotated = sum(part_annot)
                # get overall distance between n1 stop and n2 start
                dist = self.G.nodes[n2]["start"] - self.G.nodes[n1]["stop"]
                # nucleotide space between n1, n2 without annotations.
                diff = dist - annotated
                assert diff > 0
        return diff

    def get_orf_lookup(self):
        self.orf_lookup = {}
        for n in self.G.nodes():
            orf_id = self.G.nodes[n]["orf_id"]
            if self.orf_lookup.get(orf_id) is None:
                self.orf_lookup[orf_id] = set()
            self.orf_lookup[orf_id].add(n)

    def add_prism_clusters(
        self,
        prism_fh: str = None,
        tolerance: int = 100,
        chemotypes_to_consider: List[str] = None,
    ):
        if not hasattr(self, "orf_lookup"):
            self.get_orf_lookup()
        clusters = get_prism_cluster_orfs(
            prism_fh, chemotypes_to_consider=chemotypes_to_consider
        )
        for i, cluster in enumerate(clusters):
            cluster_start = cluster[0]["cluster_start"]
            cluster_stop = cluster[0]["cluster_stop"]
            cluster_chemotype = cluster[0]["cluster_chemotype"]
            contig_id = cluster[0]["contig_id"]
            try:
                contig_nodes = self.contig_to_nodes[contig_id]
            except KeyError:
                raise KeyError("Contig not found in lookup")
            for node_id in contig_nodes:
                node_props = self.G.nodes[node_id]
                if (
                    abs(node_props["start"] + tolerance) < cluster_start
                    or abs(node_props["stop"] - tolerance) > cluster_stop
                ):
                    continue
                self.G.nodes[node_id]["cluster_chemotype"] = cluster_chemotype
                self.G.nodes[node_id]["cluster_node_cat"] = "core"

    def identify_peripheral_orfs(self, tolerance: int = 10000):
        for contig_id, node_ids in self.contig_to_nodes.items():
            node_ids = sorted(node_ids, key=lambda x: self.G.nodes[x]["start"])
            for idx, n1 in enumerate(node_ids):
                for n2 in node_ids[idx + 1 :]:
                    diff = self.get_diff(n1, n2, method=self.diff_method)
                    if diff > tolerance:
                        break
                    elif (
                        self.G.nodes[n1]["cluster_node_cat"] == None
                        and self.G.nodes[n2]["cluster_node_cat"] == "core"
                    ):
                        self.G.nodes[n1]["cluster_node_cat"] = "peripheral"
                    elif (
                        self.G.nodes[n1]["cluster_node_cat"] == "core"
                        and self.G.nodes[n2]["cluster_node_cat"] == None
                    ):
                        self.G.nodes[n2]["cluster_node_cat"] = "peripheral"

    def add_antismash_clusters(
        self,
        antismash_fh: str = None,
        tolerance: int = 100,
        add_peripheral: bool = True,
        chemotypes_to_consider: List[str] = None,
        prioritize_prism: bool = True,
    ):
        if not hasattr(self, "orf_lookup"):
            self.get_orf_lookup()
        clusters = get_antismash_clusters(
            antismash_fh, chemotypes_to_consider=chemotypes_to_consider
        )
        for i, cluster in enumerate(clusters):
            cluster_start = cluster["cluster_start"]
            cluster_stop = cluster["cluster_stop"]
            cluster_chemotype = cluster["cluster_chemotype"]
            contig_id = cluster["contig_id"]
            peri_start = cluster["peripheral_start"]
            peri_stop = cluster["peripheral_stop"]
            try:
                contig_nodes = self.contig_to_nodes[contig_id]
            except KeyError:
                raise KeyError("Contig not found in lookup")
            peri_nodes_to_consider = []
            cluster_nodes_to_consider = []
            for node_id in contig_nodes:
                node_props = self.G.nodes[node_id]
                nstart = node_props["start"]
                nstop = node_props["stop"]
                # if outside peripheral boundaries, ignore
                if (
                    abs(nstart + tolerance) < peri_start
                    or abs(nstop - tolerance) > peri_stop
                ):
                    continue
                else:
                    peri_nodes_to_consider.append(node_id)
                    if (
                        abs(node_props["start"] + tolerance) < cluster_start
                        or abs(node_props["stop"] - tolerance) > cluster_stop
                    ):
                        continue
                    if prioritize_prism:
                        if node_props.get("cluster_node_cat", "") == "core":
                            # do not add cluster if any node overlaps
                            # with a prism core.
                            break
                        else:
                            cluster_nodes_to_consider.append(node_id)
                    else:
                        raise NotImplementedError
            if prioritize_prism:
                for node_id in cluster_nodes_to_consider:
                    self.G.nodes[node_id][
                        "cluster_chemotype"
                    ] = cluster_chemotype
                    self.G.nodes[node_id]["cluster_node_cat"] = "core"
            if add_peripheral:
                for node_id in set(peri_nodes_to_consider) - set(
                    cluster_nodes_to_consider
                ):
                    node_props = self.G.nodes[node_id]
                    if node_props.get("cluster_node_cat", "") == "core":
                        continue
                    else:
                        self.G.nodes[node_id][
                            "cluster_node_cat"
                        ] = "peripheral"

    def get_biosynthetic_windows(self, size: int = 200, step: int = 125):
        windows_to_keep = []
        core_orfs = self.core_orfs
        for contig_id, node_ids in self.contig_to_nodes.items():
            if len(core_orfs.intersection(node_ids)) == 0:
                continue
            node_ids = sorted(node_ids, key=lambda x: self.G.nodes[x]["start"])
            for window in mit.windowed(node_ids, n=size, step=step):
                window = [
                    n for n in window if n != None
                ]  # remove None padding.
                if len(set(window) & core_orfs) > 0:
                    windows_to_keep.append(window)
        return windows_to_keep

    def get_contig_windows(self, size: int = 200, step: int = 125):
        windows = []
        for contig_id, node_ids in self.contig_to_nodes.items():
            node_ids = sorted(node_ids, key=lambda x: self.G.nodes[x]["start"])
            for window_idx, window in enumerate(
                mit.windowed(node_ids, n=size, step=step)
            ):
                window = [
                    n for n in window if n != None
                ]  # remove None padding.
                if window_idx > 0 and len(window) < (size - step):
                    continue
                windows.append(window)
        return windows


def get_graph_from_orfs(
    orfs: List[OrfData],
    save_fp: str,
    orf_tolerance: int = 10000,
    orf_edge_method: Literal["raw", "unannotated"] = "raw",
    force_perfect_domain_edges: bool = False,
    include_domains: bool = True,
    prism_fh: str = None,
    cluster_orf_tolerance=100,
    antismash_fh: str = None,
    identify_cluster_peripheral: bool = True,
    prism_chemotypes_to_consider: List[str] = None,
):
    graph = OrfGraph()
    graph.add_orfs(
        orfs=orfs,
        tolerance=orf_tolerance,
        diff_method=orf_edge_method,
        force_perfect_domain_edges=force_perfect_domain_edges,
        include_domains=include_domains,
    )
    if prism_fh is not None:
        try:
            graph.add_prism_clusters(
                prism_fh=prism_fh,
                tolerance=cluster_orf_tolerance,
                chemotypes_to_consider=prism_chemotypes_to_consider,
            )
            if identify_cluster_peripheral:
                graph.identify_peripheral_orfs(tolerance=10000)
        except KeyError:
            return None
    if antismash_fh is not None:
        try:
            graph.add_antismash_clusters(
                antismash_fh=antismash_fh,
                tolerance=cluster_orf_tolerance,
                add_peripheral=True,
                chemotypes_to_consider=prism_chemotypes_to_consider,
                prioritize_prism=True,
            )
        except KeyError:
            return None
    if save_fp:
        pickle.dump(graph, open(save_fp, "wb"))
    else:
        return graph


def check_window_quality(window_G: nx.Graph, node_threshold: int = 50):
    # bgc_subwindows
    nodes = sorted(window_G.nodes())
    if len(nodes) >= node_threshold:
        return "quality"
    else:
        return "non_quality"


def get_biosynthetic_windows_from_orfs(
    orfs: List[OrfData],
    save_dir: str,
    prism_fh: str,
    antismash_fh: str = None,
    window_size: int = 200,
    window_overlap: int = 75,
    orf_tolerance: int = 10000,
    orf_edge_method: Literal["raw", "unannotated"] = "raw",
    force_perfect_domain_edges: bool = False,
    include_domains: bool = False,
    cluster_orf_tolerance=100,
    cache_window_orfs: bool = True,
    node_window_threshold: int = 50,
    prism_chemotypes_to_consider: List[str] = None,
):
    graph = OrfGraph()
    graph.add_orfs(
        orfs=orfs,
        tolerance=orf_tolerance,
        diff_method=orf_edge_method,
        force_perfect_domain_edges=force_perfect_domain_edges,
        include_domains=include_domains,
    )
    try:
        graph.add_prism_clusters(
            prism_fh=prism_fh,
            tolerance=cluster_orf_tolerance,
            chemotypes_to_consider=prism_chemotypes_to_consider,
        )
    except KeyError:
        print(
            "There is an issue with the contig lookup "
            f"for this entry. {prism_fh}"
        )
        return None
    graph.identify_peripheral_orfs(tolerance=10000)
    if antismash_fh is not None:
        try:
            graph.add_antismash_clusters(
                antismash_fh=antismash_fh,
                tolerance=cluster_orf_tolerance,
                add_peripheral=True,
                chemotypes_to_consider=prism_chemotypes_to_consider,
                prioritize_prism=True,
            )
        except KeyError:
            print(
                "There is an issue with the contig lookup "
                f"for this entry. {antismash_fh}"
            )
            return None
    window_step = window_size - window_overlap
    assert window_step > 0
    windows = graph.get_biosynthetic_windows(
        size=window_size, step=window_step
    )
    window_orfs = {}
    window_stats = {}
    for window_idx, window in enumerate(windows, start=1):
        save_fp = os.path.join(save_dir, f"{window_idx}.nx.pkl")
        subG = graph.G.subgraph(window).copy()
        quality = check_window_quality(
            subG, node_threshold=node_window_threshold
        )
        window_chemotypes = list(
            set(
                [
                    dat.get("cluster_chemotype")
                    for n, dat in subG.nodes(data=True)
                    if dat.get("cluster_chemotype") is not None
                ]
            )
        )
        window_stats[window_idx] = {
            "contig_quality": quality,
            "window_chemotypes": window_chemotypes,
        }
        pickle.dump(
            subG, open(save_fp, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )
        if cache_window_orfs:
            for n in subG.nodes():
                node_props = subG.nodes[n]
                orf_id = node_props["orf_id"]
                node_start = node_props["start"]
                node_stop = node_props["stop"]
                if (
                    include_domains is False
                    or node_props["node_type"] == "orf"
                ):
                    window_orfs[orf_id] = {
                        "start": node_start,
                        "stop": node_stop,
                        "protein_id": node_props["protein_id"],
                    }
                else:
                    # look up orf start and stop for given domain
                    window_orfs[orf_id] = {
                        "start": graph.orf_to_domains[orf_id]["orf_start"],
                        "stop": graph.orf_to_domains[orf_id]["orf_stop"],
                    }
    json_fp = os.path.join(save_dir, "window_info.json")
    json.dump(window_stats, open(json_fp, "w"))
    return window_orfs


def create_tensor_from_graph(
    graph: nx.Graph, x: torch.tensor, node_to_idx: Dict[int, int]
):
    num_edges = graph.number_of_edges()
    edge_index = np.zeros((2, num_edges), dtype=int)
    edge_attr = []
    for edge_idx, (n1, n2, prop) in enumerate(graph.edges(data=True)):
        edge_index[0][edge_idx] = node_to_idx[n1]
        edge_index[1][edge_idx] = node_to_idx[n2]
        edge_attr.append([prop["score"]])
    # create tensors
    # Data requires edge index format as LongTensor of [2, num_edges] shape
    edge_index = torch.LongTensor(edge_index)
    score_tensor = torch.tensor(edge_attr)
    # Data requires [num_edges, num_edge_features] shape for edge_attr
    datapoint = Data(x=x, edge_index=edge_index, edge_attr=score_tensor)
    return datapoint
