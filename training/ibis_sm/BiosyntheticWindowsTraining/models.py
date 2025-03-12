# from HadronGraph.GenomicOrf.BiosyntheticWindows.GATGraphormer.NonDynamic import constants
import json
import os
import pickle
from typing import Callable, Optional

import torch
from torch.nn import ModuleDict

from omnicons import datdir
from omnicons.lightning.GraphModelForMultiTask import (
    GraphModelForMultiTaskLightning,
)
from omnicons.metrics import ClassificationMetrics
from omnicons.optimizers.preconfigured import get_deepspeed_adamw
from training.ibis_sm.BiosyntheticWindowsTraining.data_modules import (
    BiosyntheticWindowMultiDataModule,
)


def get_cluster_chemotype_dict(cluster_chemotype_fp: str = None) -> dict:
    from omnicons import datdir

    cluster_chemotype_fp = (
        cluster_chemotype_fp
        if cluster_chemotype_fp is not None
        else os.path.join(
            datdir, "biosyn_windows" "label_files", "cluster_chemotype.json"
        )
    )
    return json.load(open(cluster_chemotype_fp, "r"))


def get_gnn(
    config_dir: str,
    embedding_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
):
    from omnicons.configs.GNNConfigs import GATConfig

    gnn_config = GATConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embedding_dim,
        edge_dim=1,
        dropout=dropout,
    )
    gnn_config.save_to(f"{config_dir}/gnn_config.json")
    return gnn_config


def get_node_encoder(
    config_dir: str,
    input_dim=1024,
    output_dim=256,
    dropout=0.1,
    num_layers=1,
):
    from omnicons.configs.EncoderConfigs import MLPEncoderConfig

    node_encoder_config = MLPEncoderConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout,
        num_layers=num_layers,
    )
    node_encoder_config.save_to(f"{config_dir}/node_encoder_config.json")
    return node_encoder_config


def get_transformer(
    config_dir: str,
    embedding_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    mlp_dropout: float = 0.1,
):
    from omnicons.configs.TransformerConfigs import GraphormerConfig

    transformer_config = GraphormerConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embedding_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        mlp_dropout=mlp_dropout,
    )
    transformer_config.save_to(f"{config_dir}/transformer_config.json")
    return transformer_config


def get_graph_pooler(
    config_dir: str,
    embedding_dim: int = 256,
):
    # With this setup, node regression pooler is the same as node classification pooler (I think.)
    from omnicons.configs.GraphPoolerConfigs import NodeClsPoolerConfig

    graph_pooler_config = NodeClsPoolerConfig(hidden_channels=embedding_dim)
    graph_pooler_config.save_to(f"{config_dir}/graph_pooler_config.json")
    return graph_pooler_config


def get_heads(
    config_dir,
    weights,
    clust_chemotypes,
    head_hidden_dropout_prob=0.1,
    embedding_dim=256,
):
    from omnicons.configs.HeadConfigs import NodeClsTaskHeadConfig

    heads = {}
    heads["clust_chemotype"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=head_hidden_dropout_prob,
        num_labels=len(clust_chemotypes),
        class_weight=weights["cluster_chemotype"],
        multi_label=False,
        analyze_inputs=["a", "b"],
    )
    heads["boundary_cat"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=head_hidden_dropout_prob,
        num_labels=1,  # ignored for binary tasks anyway?
        class_weight=weights["boundary_cat"],
        binary=True,
        multi_label=False,
        analyze_inputs=["a", "b"],
    )
    for head_name, head_config in heads.items():
        head_config.save_to(f"{config_dir}/{head_name}_head_config.json")
    return heads


def get_model(
    dm: BiosyntheticWindowMultiDataModule = None,
    weights_fp: str = os.path.join(
        datdir, "biosyn_windows", "biosyn_windows_weights.pkl"
    ),
    config_dir: str = os.path.join(
        datdir, "training", "biosyn_windows", "configs"
    ),
    checkpoint_path: Optional[str] = None,
    node_embedding_dim: int = 256,
    node_encoder_dropout: float = 0.1,
    node_encoder_num_layers: int = 1,
    gat_num_layers: int = 4,
    gat_num_heads: int = 4,
    gat_dropout: float = 0.1,
    transformer_num_layers: int = 4,
    transformer_num_heads: int = 4,
    transformer_attention_dropout: float = 0.1,
    transformer_mlp_dropout: float = 0.1,
    head_dropout: float = 0.1,
    optimizer: Callable = get_deepspeed_adamw,
):
    if weights_fp is not None:
        class_weights = pickle.load(open(weights_fp, "rb"))
    elif dm is not None:
        class_weights = dm.compute_class_weights()
    else:
        raise ValueError("Must provide either weights_fp or DataModule")
    # class dicts
    clust_chemotypes = get_cluster_chemotype_dict()
    # model setup
    node_encoder_config = get_node_encoder(
        config_dir=config_dir,
        input_dim=1024,
        output_dim=node_embedding_dim,
        dropout=node_encoder_dropout,
        num_layers=node_encoder_num_layers,
    )
    gnn_config = get_gnn(
        config_dir=config_dir,
        embedding_dim=node_embedding_dim,
        num_layers=gat_num_layers,
        num_heads=gat_num_heads,
        dropout=gat_dropout,
    )
    transformer_config = get_transformer(
        config_dir=config_dir,
        embedding_dim=node_embedding_dim,
        num_layers=transformer_num_layers,
        num_heads=transformer_num_heads,
        dropout=head_dropout,
        attention_dropout=transformer_attention_dropout,
        mlp_dropout=transformer_mlp_dropout,
    )
    graph_pooler_config = get_graph_pooler(
        config_dir=config_dir,
        embedding_dim=node_embedding_dim,
    )
    heads = get_heads(
        config_dir=config_dir,
        weights=class_weights,
        clust_chemotypes=clust_chemotypes,
    )
    # Metrics
    train_metrics = ModuleDict(
        {
            "clust_chemotype___a": ClassificationMetrics.get(
                name="clust_chemotype___a_train",
                num_classes=len(clust_chemotypes),
                task="multiclass",
                ignore_index=-100,
                average_strategy="weighted",
            ),
            "boundary_cat___a": ClassificationMetrics.get(
                name="boundary_cat___a_train", task="binary", ignore_index=-100
            ),
            "clust_chemotype___b": ClassificationMetrics.get(
                name="clust_chemotype___b_train",
                num_classes=len(clust_chemotypes),
                task="multiclass",
                ignore_index=-100,
                average_strategy="weighted",
            ),
            "boundary_cat___b": ClassificationMetrics.get(
                name="boundary_cat___b_train", task="binary", ignore_index=-100
            ),
        }
    )
    val_metrics = ModuleDict(
        {
            "clust_chemotype___a": ClassificationMetrics.get(
                name="clust_chemotype___a_val",
                num_classes=len(clust_chemotypes),
                task="multiclass",
                ignore_index=-100,
                average_strategy="weighted",
            ),
            "boundary_cat___a": ClassificationMetrics.get(
                name="boundary_cat___a_val", task="binary", ignore_index=-100
            ),
            "clust_chemotype___b": ClassificationMetrics.get(
                name="clust_chemotype___b_val",
                num_classes=len(clust_chemotypes),
                task="multiclass",
                ignore_index=-100,
                average_strategy="weighted",
            ),
            "boundary_cat___b": ClassificationMetrics.get(
                name="boundary_cat___b_val", task="binary", ignore_index=-100
            ),
        }
    )
    # Instantiate a PyTorch Lightning Module
    model = GraphModelForMultiTaskLightning(
        node_encoder_config=node_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        graph_pooler_config=graph_pooler_config,
        heads=heads,
        optimizer_fn=optimizer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        inputs=["a", "b"],
    )
    clust_chemotypes = get_cluster_chemotype_dict()
    # load checkpoint
    if checkpoint_path != None:
        states = torch.load(checkpoint_path)
        model.load_state_dict(states["state_dict"], strict=True)
    return model
