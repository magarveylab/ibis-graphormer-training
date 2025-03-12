import os
import pickle
from functools import partial
from glob import glob
from multiprocessing import Pool
from typing import Literal, Tuple

import pandas as pd
from tqdm import tqdm

from omnicons import datdir
from omnicons.data_generators.genomic.utils import get_orflist_from_fh
from omnicons.graph_converters.homogeneous.genome_graph import OrfGraph


def create_graph_from_csv_fp(
    # path to genome window csv file and fasta amino acid file.
    fhs: Tuple[str, str],
    graph_dir: str,  # directory to save graphs to
    orf_tolerance: int = 10000,
    orf_edge_method: Literal["raw", "unannotated"] = "raw",
    force_perfect_domain_edges: bool = False,
    include_domains: bool = False,
):
    csv_fp, faa_fh = fhs
    orfs = get_orflist_from_fh(faa_fh=faa_fh)
    graph = OrfGraph()
    graph.add_orfs(
        orfs=orfs,
        tolerance=orf_tolerance,
        diff_method=orf_edge_method,
        force_perfect_domain_edges=force_perfect_domain_edges,
        include_domains=include_domains,
    )
    df = pd.read_csv(csv_fp)
    basename = os.path.basename(csv_fp).replace(".csv", "")
    save_dir = os.path.join(graph_dir, basename)
    os.makedirs(save_dir, exist_ok=True)
    # replace pd.na with None for compatibility moving forward.
    chemotype_lkp = {
        a: b if pd.notna(b) else None
        for a, b in zip(df["node_id"], df["chemotype"])
    }
    peripheral_lkp = {
        a: b if pd.notna(b) else None
        for a, b in zip(df["node_id"], df["node_category"])
    }
    for window_idx, sub_df in df.groupby("window_id"):
        node_ids = sub_df["node_id"].tolist()
        subG = graph.G.subgraph(node_ids).copy()
        save_fp = os.path.join(save_dir, f"{window_idx}.nx.pkl")
        for node_id in node_ids:
            subG.nodes[node_id]["cluster_chemotype"] = chemotype_lkp[node_id]
            subG.nodes[node_id]["cluster_node_cat"] = peripheral_lkp[node_id]
        pickle.dump(
            subG, open(save_fp, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )


if __name__ == "__main__":
    all_fhs = glob(
        os.path.join(datdir, "biosyn_windows", "genome_csvs", "*.csv")
    )
    faa_fhs = []  # populate this with a list of all corresponding fasta files.
    graph_dir = ""  # path to save graphs to.
    pool = Pool(30)
    F = partial(create_graph_from_csv_fp, graph_dir=graph_dir)
    process = pool.imap_unordered(F, zip(all_fhs, faa_fhs))
    [p for p in tqdm(process, total=len(all_fhs))]
