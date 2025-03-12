import ast
import json
import os
import pickle
from glob import glob
from typing import Dict, List, Optional, TypedDict, Union

import pandas as pd
import torch
import xxhash
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from omnicons import helpers

prism_to_common_chemotypes = {
    "AMINOCOUMARIN": "Aminocoumarin",
    "PHENAZINE": "Phenazine",
    "PHOSPHONATE": "Phosphonate",
    "BETA_LACTAM": "BetaLactam",
    "PHENYL": "Phenyl",
    "NIS_SYNTHASE": "NRPS-IndependentSiderophore",
    "NONRIBOSOMAL_PEPTIDE": "NonRibosomalPeptide",
    "BUTYROLACTONE": "Butyrolactone",
    "FURAN": "Furan",
    "ANTIMETABOLITE": "Antimetabolite",
    "BACTERIOCIN": "Bacteriocin",
    "ECTOINE": "Ectoine",
    "INDOLE": "Indole",
    "HOMOSERINE_LACTONE": "HomoserineLactone",
    "BISINDOLE": "Bisindole",
    "TYPE_II_POLYKETIDE": "TypeIIPolyketide",
    "AMINOGLYCOSIDE": "Aminoglycoside",
    "RIBOSOMAL": "Ripp",
    "PHOSPHOGLYCOLIPID": "Phosphoglycolipid",
    "CYCLODIPEPTIDE": "Cyclodipeptide",
    "HAPALINDOLE": "Hapalindole",
    "TERPENE": "Terpene",
    "NUCLEOSIDE": "Nucleoside",
    "ALKALOID": "Alkaloid",
    "TYPE_I_POLYKETIDE": "TypeIPolyketide",
    "RESORCINOL": "Resorcinol",
    "ARYL_POLYENE": "ArylPolyene",
    "MELANIN": "Melanin",
    "LINCOSAMIDE": "Lincosamide",
    "CHLORAMPHENICOL": "Chloramphenicol",
    "ISOPROPYLSTILBENE": "Stilbene",
    "ENTEROCIN": "Enterocin",
    "ENEDIYNE_10_MEMBERED": "Enediyne",
    "ENEDIYNE_9_MEMBERED": "Enediyne",
    "ISONITRILE": "Isonitrile",
}
all_prism_chemotypes = list(set(prism_to_common_chemotypes.values()))
all_prism_chemotypes_w_hybrid = all_prism_chemotypes + ["PKSNRPSHybrid"]

prism_orf_to_common_chemotypes = {
    "resistance": [
        "Resistance"
    ],  # Not a native bear type, but still want to predict
    "modification": ["Modifying"],  # Not a native bear type
    "pks": ["TypeIPolyketide"],
    "nrps": ["NonRibosomalPeptide"],
    "tailoring": [
        "Tailoring"
    ],  # Not a native bear type, but still want to predict
    "nucleoside": ["Nucleoside"],
    "bacteriocin": ["Bacteriocin"],
    "ribosomal": ["Ripp"],
    "isonitrile": ["Isonitrile"],
    "nis_synthase": ["NRPS-IndependentSiderophore"],
    "aminoglycoside": ["Aminoglycoside"],
    "regulator": [
        "Regulator"
    ],  # Not a native bear type, but still want to predict
    "phosphonate": ["Phosphonate"],
    "hybrid": ["TypeIPolyketide", "NonRibosomalPeptide"],
    "beta_lactam": ["BetaLactam"],
    "bisindole": ["Bisindole"],
    "prerequisite": ["Prerequisite"],  # Not a native bear type
    "type_ii_pks": ["TypeIIPolyketide"],
    "cyclodipeptide": ["Cyclodipeptide"],
    "antimetabolite": ["Antimetabolite"],
    "sugar": ["Sugar"],  # Not a native bear type, but still want to predict
    "terpene": ["Terpene"],
    "aminocoumarin": ["Aminocoumarin"],
    "primary_metabolite": [
        "PrimaryMetabolite"
    ],  # Not a native bear type, but still want to predict
}


class DomainData(TypedDict):
    contig_id: Union[int, str]
    orf_id: Union[
        int, str
    ]  # of the form <contig_name>_<orf_enum> as presented by prodigal or cactus orf_id
    protein_id: int  # hashed protein sequence
    start: int
    stop: int
    domain_type: str
    embedding: torch.tensor


class OrfData(TypedDict):
    contig_id: Union[int, str]
    orf_id: Union[
        int, str
    ]  # of the form <contig_name>_<orf_enum> as presented by prodigal or cactus orf_id
    protein_id: int  # hashed protein sequence
    start: int
    stop: int
    domains: List[DomainData]
    embedding: torch.tensor
    ec_number: Optional[str]
    ec_homology_score: Optional[float]


class PrismOrfData(TypedDict):
    contig_id: Union[int, str]
    orf_id: Union[
        int, str
    ]  # of the form <contig_name>_<orf_enum> as presented by prodigal or cactus orf_id
    start: int
    stop: int
    cluster_start: int
    cluster_stop: int
    cluster_chemotype: str


class AntismashClusterData(TypedDict):
    contig_id: Union[int, str]
    cluster_start: int
    cluster_stop: int
    peripheral_start: int
    peripheral_stop: int
    cluster_chemotype: str


def get_antismash_chemotype_lookup():
    as_ctype_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "dat", "antismash_to_common.csv"
        )
    )
    antismash_ctype_lookup = {}
    for x in as_ctype_df.to_dict("records"):
        bc = x["bear_chemotype"]
        if pd.isna(bc):
            antismash_ctype_lookup[x["antismash_chemotype"]] = None
        else:
            antismash_ctype_lookup[x["antismash_chemotype"]] = bc
    return antismash_ctype_lookup


antismash_ctype_lookup = get_antismash_chemotype_lookup()

universal_chemotypes_w_hybrid = list(
    set(all_prism_chemotypes + ["PKSNRPSHybrid"])
    | set(antismash_ctype_lookup.values()) - set([None])
)

chemotypes_to_use = json.load(
    open(os.path.join(os.path.dirname(__file__), "dat", "chemotypes.json"))
)


def subset_genomes(
    subset_dir,
    df_fp,
    genome_column: str = "working_submission_fh",
    chemotype_column: str = "chemotypes",
    split_proportion: float = 0.1,
):
    if all(
        [
            os.path.isfile(f"{subset_dir}/{x}.csv")
            for x in ["train", "val", "test"]
        ]
    ):
        print("splits already created.")
    else:
        chemotype_to_genomes = {}
        selected_genomes = {"train": set(), "val": set(), "test": set()}
        cache = set()
        data = pd.read_csv(df_fp).to_dict("records")
        print(f"Genomes: {len(data)}")
        # organize genomes by chemotypes
        for rec in tqdm(data):
            for chemotype_key in ast.literal_eval(rec[chemotype_column]):
                chemotypes = chemotype_key.split("__")
                for chemotype in chemotypes:
                    if chemotype not in chemotype_to_genomes:
                        chemotype_to_genomes[chemotype] = set()
            chemotype_to_genomes[chemotype].add(rec[genome_column])
        # train test split by chemotype - requires at least 3 samples per category.
        chemotype_to_genomes = [
            x
            for x in sorted(
                chemotype_to_genomes.items(), key=lambda x: len(x[1])
            )
            if len(x[1]) >= 3
        ]
        for chemotype, genomes in chemotype_to_genomes:
            genomes = list(genomes - cache)
            genomes = sorted(genomes)
            train, test = train_test_split(
                genomes, test_size=split_proportion, random_state=42
            )
            train, val = train_test_split(
                train, test_size=split_proportion, random_state=42
            )
            cache.update(genomes)
            selected_genomes["train"].update(train)
            selected_genomes["val"].update(val)
            selected_genomes["test"].update(test)
        # subset metadata for selected genoems.
        helpers.create_dir(subset_dir)
        final = {"train": [], "val": [], "test": []}
        for split in ["train", "val", "test"]:
            for rec in tqdm(data):
                if rec[genome_column] in selected_genomes[split]:
                    final[split].append(rec)
        # export subsets
        for split, recs in final.items():
            print(f"{split} genomes: {len(recs)}")
            pd.DataFrame(recs).to_csv(f"{subset_dir}/{split}.csv", index=None)


def embed_file_lookup(dirname):
    out = {}
    for fh in glob(os.path.join(dirname, "*.pkl")):
        basename = os.path.basename(fh)
        results = pickle.load(open(fh, "rb"))
        for i, result in enumerate(results):
            out[result["protein_id"]] = {"file": basename, "position": i}
    return out


def get_prism_cluster_orfs(
    prism_fh,
    for_cluster_only: bool = False,
    chemotypes_to_consider: List[str] = None,
) -> List[List[PrismOrfData]]:
    if chemotypes_to_consider is None:
        chemotypes_to_consider = universal_chemotypes_w_hybrid
    clusters = json.load(open(prism_fh, "r"))["prism_results"]["clusters"]
    cleaned_clusters = []
    for cluster in clusters:
        contig_name = cluster["contig_name"].split(" ")[0]
        cluster_chemotypes = [
            prism_to_common_chemotypes.get(x) for x in cluster.get("family")
        ]
        cluster_chemotypes = [x for x in cluster_chemotypes if x is not None]
        cluster_chemotypes = list(
            set(cluster_chemotypes).intersection(chemotypes_to_consider)
        )
        if cluster_chemotypes == []:
            continue
        elif len(cluster_chemotypes) == 1:
            cluster_chemotype = cluster_chemotypes[0]
        elif (
            len(
                set(cluster_chemotypes).intersection(
                    ["NonRibosomalPeptide", "TypeIPolyketide"]
                )
            )
            == 2
        ):
            cluster_chemotype = "PKSNRPSHybrid"
        else:
            continue
        cluster_orfs = []
        for orf in cluster["orfs"]:
            # genomes have prodigal annotations only - and so will production applications.
            # note that this may remove a small number of prism annotations.
            if orf["mode"] != "PRODIGAL":
                continue
            if len(orf["domains"]) == 0:
                continue  # ensure prism actually annotates it.
            # convert ambiguous prefix to contig name + orf number (i.e. 'orf_123' -> '<contig_name>_123)
            orf_name = "_".join([contig_name, orf["name"].split("_")[1]])
            # orf_chemotypes = prism_orf_to_bear_chemotypes.get(orf.get('type'))
            cluster_orf = {
                "contig_id": orf_name.split("_")[0],
                "orf_id": orf_name,
                "start": int(orf["start"])
                + 1,  # for some reason, there is a +1 disparity between prism results and prodigal.
                "stop": int(orf["stop"]),
                "cluster_start": int(cluster["start"]),
                "cluster_stop": int(cluster["end"]),
                "cluster_chemotype": cluster_chemotype,
            }
            # similar role as get_orflist_from_fh()
            if for_cluster_only:
                import xxhash

                seq = orf["sequence"].replace("*", "")
                prot_id = xxhash.xxh32(seq).intdigest()
                cluster_orf["protein_id"] = prot_id
                cluster_orf["contig_id"] = contig_name
                cluster_orf["domains"] = []
                cluster_orf["embedding"] = None
            cluster_orfs.append(cluster_orf)
        if cluster_orfs == []:
            continue
        cleaned_clusters.append(cluster_orfs)
    return cleaned_clusters


def _resolve_single_antismash_cluster(
    protocluster: dict, contig_name: str, return_all: bool = True
) -> Union[None, AntismashClusterData]:
    bgc_start = protocluster["core_start"]
    bgc_stop = protocluster["core_end"]
    periph_start = protocluster["start"]
    periph_stop = protocluster["end"]
    bgc_raw_chemotype = protocluster["product"]
    bgc_bear_chemotype = antismash_ctype_lookup.get(bgc_raw_chemotype)
    if bgc_bear_chemotype is None:
        if return_all:
            return {
                "contig_id": contig_name,
                "peripheral_start": periph_start,
                "cluster_start": bgc_start,
                "cluster_stop": bgc_stop,
                "peripheral_stop": periph_stop,
                "cluster_chemotype": bgc_bear_chemotype,
                "antismash_chemotype": bgc_raw_chemotype,
            }
        else:
            return None
    else:
        cluster_orf = {
            "contig_id": contig_name,
            "peripheral_start": periph_start,
            "cluster_start": bgc_start,
            "cluster_stop": bgc_stop,
            "peripheral_stop": periph_stop,
            "cluster_chemotype": bgc_bear_chemotype,
            "antismash_chemotype": bgc_raw_chemotype,
        }
        return cluster_orf


def _resolve_hybrid_antismash_cluster(
    protoclusters: List[Dict[int, dict]], contig_name: str
) -> Union[None, AntismashClusterData]:
    starts = []
    stops = []
    pstarts = []
    pstops = []
    ctypes = []
    bear_ctypes = set()
    # assign metadata
    for protocluster in protoclusters:
        reg_start = protocluster["core_start"]
        starts.append(reg_start)
        reg_stop = protocluster["core_end"]
        pstarts.append(protocluster["start"])
        pstops.append(protocluster["end"])
        stops.append(reg_stop)
        reg_len = reg_stop - reg_start
        protocluster["core_length"] = reg_len
        reg_ctype = protocluster["product"]
        ctypes.append(reg_ctype)
        bear_ctype = antismash_ctype_lookup.get(reg_ctype)
        bear_ctypes.add(bear_ctype)
    if (
        len(
            bear_ctypes.intersection(
                ["TypeIPolyketide", "NonRibosomalPeptide"]
            )
        )
        == 2
    ):
        # should be a hybrid NRPS-PKS
        bgc_start = min(starts)
        periph_start = min(pstarts)
        bgc_stop = max(stops)
        periph_stop = max(pstops)
        bear_chemotype = "PKSNRPSHybrid"
        cluster_orf = {
            "contig_id": contig_name,
            "peripheral_start": periph_start,
            "cluster_start": bgc_start,
            "cluster_stop": bgc_stop,
            "peripheral_stop": periph_stop,
            "cluster_chemotype": bear_chemotype,
            "antismash_chemotype": "T1PKS-NRPS",
        }
        return cluster_orf
    else:
        return None


def get_antismash_clusters(
    antismash_fp: str, chemotypes_to_consider: List[str] = None
) -> List[AntismashClusterData]:
    if chemotypes_to_consider is None:
        chemotypes_to_consider = universal_chemotypes_w_hybrid
    return_all = True if chemotypes_to_consider == "all" else False
    records = json.load(open(antismash_fp, "r"))["records"]
    cleaned_clusters = []
    for record in records:
        contig_name = record["name"]
        areas = record["areas"]
        for area in areas:
            candidates = area["candidates"]
            protocs = area["protoclusters"]
            for candidate in candidates:
                protocs_to_consider = candidate["protoclusters"]
                kind = candidate["kind"]
                # neighbouring are always accompanied by the associated singles.
                # treat neighbouring as the composite singles. For interleaved or hybrid, prioritize those.
                if kind == "neighbouring":
                    continue
                elif kind == "chemical_hybrid":
                    protocss = [
                        protocs[str(ptc)] for ptc in protocs_to_consider
                    ]
                    cluster_result = _resolve_hybrid_antismash_cluster(
                        protoclusters=protocss, contig_name=contig_name
                    )
                elif kind == "interleaved":
                    overlap = []
                    for ptc1 in protocs_to_consider:
                        for ptc2 in protocs_to_consider[ptc1 + 1 :]:
                            overlap.append(
                                protocs[str(ptc2)]["core_start"]
                                < protocs[str(ptc1)]["core_end"]
                            )
                    if any(overlap):
                        # cluster cores overlap - consider hybrid
                        protocss = [
                            protocs[str(ptc)] for ptc in protocs_to_consider
                        ]
                        cluster_result = _resolve_hybrid_antismash_cluster(
                            protoclusters=protocss, contig_name=contig_name
                        )
                    else:
                        # cluster cores don't overlap - treat as single clusters
                        for protoc_idx in protocs_to_consider:
                            protoc = protocs[str(protoc_idx)]
                            cluster_result = _resolve_single_antismash_cluster(
                                protocluster=protoc,
                                contig_name=contig_name,
                                return_all=return_all,
                            )
                elif kind == "single":
                    protoc = protocs[str(candidate["protoclusters"][0])]
                    cluster_result = _resolve_single_antismash_cluster(
                        protocluster=protoc,
                        contig_name=contig_name,
                        return_all=return_all,
                    )
                else:
                    raise NotImplementedError
                if cluster_result is not None:
                    if (
                        chemotypes_to_consider == "all"
                        or cluster_result["cluster_chemotype"]
                        in chemotypes_to_consider
                    ):
                        cleaned_clusters.append(cluster_result)
    return cleaned_clusters


# Returns a list of OrfData objects from a fasta file
# Embeddings are to be patched into this object later
def get_orflist_from_fh(faa_fh: str = None) -> List[OrfData]:
    records = list(SeqIO.parse(open(faa_fh, "r"), format="fasta"))
    proteins = []
    for record in records:
        seq = str(record.seq).replace("*", "")
        prot_id = xxhash.xxh32(seq).intdigest()
        name, start, stop, _, _ = record.description.split(" # ")
        proteins.append(
            {
                "contig_id": "_".join(name.split("_")[:-1]),
                "orf_id": name,
                "protein_id": prot_id,
                "start": int(start),
                "stop": int(stop),
                "domains": [],
                "embedding": None,
                "ec_number": None,
                "ec_homology_score": None,
            }
        )
    return proteins

def get_orflist_from_json(pyrodigal_json_fp:str = None, return_seqs: bool = False) -> List[OrfData]:
    records = json.load(open(pyrodigal_json_fp, 'r'))
    proteins = []
    for record in records:
        nuc_id = record['nuc_id']
        protein_id = record['protein_id']
        start = record['start']
        stop = record['end']
        orf_id = f'{nuc_id}_{start}_{stop}'
        out={
            'contig_id': nuc_id,
            'orf_id': orf_id,
            'protein_id': protein_id,
            'start': start,
            'stop': stop,
            'domains': [],
            'embedding': None,
            'ec_number': None,
            'ec_homology_score': None}
        if return_seqs:
            out['sequence']=record['sequence']
        proteins.append(out)
    return proteins
