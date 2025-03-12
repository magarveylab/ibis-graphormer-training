import os
from typing import Callable, List, Optional

import torch
from torch.nn import ModuleDict

from omnicons import datdir
from omnicons.lightning.GraphModelForMultiTask import (
    GraphModelForMultiTaskLightning,
)
from omnicons.metrics import ClassificationMetrics
from omnicons.optimizers.preconfigured import get_deepspeed_adamw
from training.ibis_sm.MibigTraining.data_modules import KFoldDataModule


def get_node_encoder(
    config_dir: str,
    input_dim: int = 1024,
    output_dim: int = 256,
    dropout: float = 0.1,
    num_layers: int = 1,
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


def get_gnn(
    config_dir: str,
    num_layers: int = 4,
    num_heads: int = 4,
    output_embed_dim: int = 256,
    edge_dim: int = 1,
    dropout: float = 0.1,
):
    from omnicons.configs.GNNConfigs import GATConfig

    gnn_config = GATConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=output_embed_dim,
        edge_dim=edge_dim,
        dropout=dropout,
    )
    gnn_config.save_to(f"{config_dir}/gnn_config.json")
    return gnn_config


def get_transformer(
    config_dir: str,
    num_layers: int = 4,
    num_heads: int = 4,
    output_embed_dim: int = 256,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    mlp_dropout: float = 0.1,
):
    from omnicons.configs.TransformerConfigs import GraphormerConfig

    transformer_config = GraphormerConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=output_embed_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        mlp_dropout=mlp_dropout,
    )
    transformer_config.save_to(f"{config_dir}/transformer_config.json")
    return transformer_config


def get_heads(
    weights: dict,
    config_dir: str,
    chemotypes_to_consider: List[str],
    suffix_options: List[str],
    hidden_size=256,
):
    from omnicons.configs.HeadConfigs import NodeClsTaskHeadConfig

    heads = {}
    for chemotype, suffix in zip(chemotypes_to_consider, suffix_options):
        heads[chemotype] = NodeClsTaskHeadConfig(
            hidden_size=hidden_size,
            hidden_dropout_prob=0.1,
            num_labels=2,
            class_weight=weights[chemotype],
            analyze_inputs=[suffix],
        )
    for head_name, head_config in heads.items():
        head_config.save_to(f"{config_dir}/{head_name}_head_config.json")
    return heads


def get_model(
    dm: KFoldDataModule,
    config_dir: str = os.path.join(datdir, "mibig_training", "configs"),
    checkpoint_path: str = os.path.join(
        datdir, "mibig_training", "pretrained_checkpoints", "ibis_sm.pt"
    ),
    chemotypes_to_consider: list = [
        "NRP",
        "Alkaloid",
        "Polyketide",
        "RiPP",
        "Saccharide",
        "Terpene",
        "Other",
    ],
    suffix_options: List[str] = ["a", "b", "c", "d", "e", "f", "g"],
    optimizer: Callable = get_deepspeed_adamw,
):
    # data module
    class_weights = dm.compute_class_weights()
    # model
    node_encoder_config = get_node_encoder(config_dir=config_dir)
    gnn_config = get_gnn(config_dir=config_dir)
    transformer_config = get_transformer(config_dir=config_dir)
    heads = get_heads(
        weights=class_weights,
        config_dir=config_dir,
        chemotypes_to_consider=chemotypes_to_consider,
        suffix_options=suffix_options,
    )
    # metrics
    metrics = {"train": ModuleDict(), "val": ModuleDict()}
    for split in ["train", "val"]:
        for chemotype, suffix in zip(chemotypes_to_consider, suffix_options):
            key = f"{chemotype}___{suffix}"
            metrics[split][key] = ClassificationMetrics.get(
                name=f"{key}_{split}", num_classes=2, task="multiclass"
            )
    # Instantiate a PyTorch Lightning Module
    model = GraphModelForMultiTaskLightning(
        node_encoder_config=node_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        heads=heads,
        optimizer_fn=optimizer,
        train_metrics=metrics["train"],
        val_metrics=metrics["val"],
        inputs=suffix_options,
    )
    # load pretrained weights
    states = torch.load(checkpoint_path)
    model.load_state_dict(states["state_dict"], strict=False)
    if checkpoint_path != None:
        states = torch.load(checkpoint_path)
        model.load_state_dict(states["state_dict"], strict=True)
    return model
