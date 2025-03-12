import os
import pickle
from functools import partial
from glob import glob
from multiprocessing import Pool
from typing import Literal, Tuple

import pandas as pd
from tqdm import tqdm

from omnicons import datdir
from omnicons.data_generators.genomic.utils import get_orflist_from_json
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
    csv_fp, faa_json_fh = fhs
    basename = os.path.basename(csv_fp).replace(".csv", "")
    orfs = get_orflist_from_json(
        pyrodigal_json_fp=faa_json_fh, return_seqs=False
    )
    graph = OrfGraph()
    graph.add_orfs(
        orfs=orfs,
        tolerance=orf_tolerance,
        diff_method=orf_edge_method,
        force_perfect_domain_edges=force_perfect_domain_edges,
        include_domains=include_domains,
    )
    save_dir = os.path.join(graph_dir, basename)
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_fp)
    # replace pd.na with None for compatibility moving forward.
    chemotype_lkp = {
        a: b if pd.notna(b) else None
        for a, b in zip(df["orf_id"], df["chemotype"])
    }
    for n, node_dat in graph.G.nodes(data=True):
        orf_id = node_dat["orf_id"]
        node_dat["cluster_chemotype"] = chemotype_lkp[orf_id]
    pickle.dump(
        graph,
        open(os.path.join(save_dir, f"{basename}.nx.pkl"), "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


if __name__ == "__main__":
    # for consistency with our tools, pyrodigal is used to predict ORFs
    # from the mibig files. If you are changing the dataset, you will
    # need to run IBIS on the fasta converted gbk file and put it
    # in this output directory
    pyrodigal_dir = os.path.join(datdir, "mibig_training", "mibig_jsons")
    csv_dir = os.path.join(datdir, "mibig_training", "mibig_csvs")
    csv_fhs = glob(os.path.join(csv_dir, "*.csv"))
    faa_fhs = [
        os.path.join(
            pyrodigal_dir, os.path.basename(x).replace(".csv", ".json")
        )
        for x in csv_fhs
    ]
    graph_dir = ""  # path to save graphs to.
    pool = Pool(30)
    F = partial(create_graph_from_csv_fp, graph_dir=graph_dir)
    process = pool.imap_unordered(F, zip(csv_fhs, faa_fhs))
    [p for p in tqdm(process, total=len(csv_fhs))]
