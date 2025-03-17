import json
import pickle

from Ibis.SecondaryMetabolismEmbedder.preprocess import BGCGraph


def create_graph_from_ibis_dir(ibis_dir: str):
    # local files needed for graph generation
    dom_emb_fp = f"{ibis_dir}/domain_embedding.pkl"
    # load domain embeddings
    dom_emb_lookup = {}
    for d in pickle.load(open(dom_emb_fp, "rb")):
        dom_emb_lookup[d["domain_id"]] = d["embedding"]
    # load protein embeddings
    prot_emb_fp = f"{ibis_dir}/protein_embedding.pkl"
    prot_emb_lookup = {}
    for p in pickle.load(open(prot_emb_fp, "rb")):
        prot_emb_lookup[p["protein_id"]] = p["embedding"]
    # load domains
    dom_pred_fp = f"{ibis_dir}/domain_predictions.json"
    domain_lookup = {}
    for p in json.load(open(dom_pred_fp)):
        protein_id = p["protein_id"]
        domain_lookup[protein_id] = []
        for r in p["regions"]:
            domain_id = r["domain_id"]
            r["embedding"] = dom_emb_lookup.get(domain_id)
            domain_lookup[protein_id].append(r)
    # load orf data
    prodigal_fp = f"{ibis_dir}/prodigal.json"
    orf_lookup = {}
    for o in json.load(open(prodigal_fp)):
        contig_id = o["contig_id"]
        contig_start = o["contig_start"]
        contig_stop = o["contig_stop"]
        orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
        protein_id = o["protein_id"]
        orf_lookup[orf_id] = {
            "contig_id": contig_id,
            "contig_start": contig_start,
            "contig_stop": contig_stop,
            "embedding": prot_emb_lookup.get(protein_id),
            "domains": domain_lookup.get(protein_id, []),
        }
    # load cluster data
    bgc_pred_fp = f"{ibis_dir}/bgc_predictions.json"
    cluster_inputs = []
    for c in json.load(open(bgc_pred_fp)):
        contig_id = c["contig_id"]
        contig_start = c["contig_start"]
        contig_stop = c["contig_stop"]
        cluster_id = f"{contig_id}_{contig_start}_{contig_stop}"
        mibig_chemotypes = c["mibig_chemotypes"]
        internal_chemotypes = c["internal_chemotypes"]
        orfs = [orf_lookup[o] for o in c["orfs"]]
        cluster_inputs.append(
            {
                "cluster_id": cluster_id,
                "mibig_chemotypes": mibig_chemotypes,
                "internal_chemotypes": internal_chemotypes,
                "orfs": orfs,
            }
        )
    return [
        {"cluster_id": c["cluster_id"], "graph": BGCGraph.from_dict(c)}
        for c in cluster_inputs
    ]
