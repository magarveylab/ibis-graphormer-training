import json
import os
from typing import Callable, Optional

import torch
from OmniconDatasets.Vocab.neo4j.genomic import get_edge_vocab, get_node_vocab
from torch.nn import ModuleDict

from omnicons import datdir
from omnicons.lightning.HeteroGraphModelForMultiTask import (
    HeteroGraphModelForMultiTaskLightning,
)
from omnicons.metrics import ClassificationMetrics
from omnicons.optimizers.preconfigured import get_deepspeed_adamw

# global parameters
node_types_with_labels = [
    "metabolite",
    "internal_chemotype",
    "mibig_chemotype",
    "domain",
]
node_types_with_embedding = ["orf", "domain_embedding"]
edge_types = [
    ("orf", "orf_to_orf", "orf"),
    ("orf", "orf_to_domain", "domain"),
    ("domain", "domain_to_domain", "domain"),
    ("domain", "domain_to_domain_embedding", "domain_embedding"),
]
labels = [
    "label_bin_1",
    "label_bin_2",
    "label_bin_3",
    "label_bin_4",
    "label_bin_5",
]
# class dict
cls_dir = os.path.join(
    datdir, "siamese_classification_training", "vocab" "partitions"
)
class_dict = json.load(open(f"{cls_dir}/class_dict.json"))


def get_node_encoders(
    vocab,
    embedding_dim: int = 256,
    node_encoder_dropout: float = 0.1,
    node_encoder_num_layers: int = 1,
):
    from omnicons.configs.EncoderConfigs import (
        MLPEncoderConfig,
        WordEncoderConfig,
    )

    node_encoders = {}
    # setup encoder for nodes with labels
    for node_type in node_types_with_labels:
        node_encoders[node_type] = WordEncoderConfig(
            num_embeddings=len(vocab[node_type]),
            embedding_dim=embedding_dim,
            extra_features=0,
            dropout=node_encoder_dropout,
            mlp_layers=node_encoder_num_layers,
        )
    # setup encoder for nodes with embedding
    for node_type in node_types_with_embedding:
        node_encoders[node_type] = MLPEncoderConfig(
            input_dim=1024,
            output_dim=embedding_dim,
            dropout=node_encoder_dropout,
            num_layers=node_encoder_num_layers,
        )
    return node_encoders


def get_edge_encoders(
    vocab,
    embedding_dim: int = 30,
    edge_encoder_dropout: float = 0.1,
    edge_encoder_num_layers: int = 1,
):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    edge_encoders = {}
    for edge_type in edge_types:
        edge_name = edge_type[1]
        edge_encoders[edge_name] = WordEncoderConfig(
            num_embeddings=len(vocab[edge_name]),
            embedding_dim=embedding_dim,
            extra_features=0,
            dropout=edge_encoder_dropout,
            mlp_layers=edge_encoder_num_layers,
        )
    return edge_encoders


def get_edge_type_encoder(
    embedding_dim: int = 30,
    edge_type_dropout: float = 0.1,
    edge_type_num_layers: int = 1,
):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    return WordEncoderConfig(
        num_embeddings=len(edge_types),
        embedding_dim=embedding_dim,
        extra_features=embedding_dim,
        dropout=edge_type_dropout,
        mlp_layers=edge_type_num_layers,
    )


def get_gnn(
    node_embedding_dim: int = 256,
    edge_embedding_dim: int = 30,
    gat_num_layers: int = 4,
    gat_num_heads: int = 4,
    gat_dropout: float = 0.1,
):
    from omnicons.configs.GNNConfigs import GATConfig

    gnn_config = GATConfig(
        num_layers=gat_num_layers,
        num_heads=gat_num_heads,
        embed_dim=node_embedding_dim,
        edge_dim=edge_embedding_dim,
        dropout=gat_dropout,
    )
    return gnn_config


def get_transformer(
    embedding_dim: int = 256,
    transformer_num_layers: int = 4,
    transformer_num_heads: int = 4,
    transformer_dropout: float = 0.1,
    transformer_attention_dropout: float = 0.1,
    transformer_mlp_dropout: float = 0.1,
):
    from omnicons.configs.TransformerConfigs import GraphormerConfig

    transformer_config = GraphormerConfig(
        num_layers=transformer_num_layers,
        num_heads=transformer_num_heads,
        embed_dim=embedding_dim,
        dropout=transformer_dropout,
        attention_dropout=transformer_attention_dropout,
        mlp_dropout=transformer_mlp_dropout,
    )
    return transformer_config


def get_graph_pooler(embedding_dim: int = 256):
    from omnicons.configs.GraphPoolerConfigs import HeteroNodeClsPoolerConfig

    graph_pooler_config = HeteroNodeClsPoolerConfig(
        node_type="metabolite", index_selector=1, hidden_channels=embedding_dim
    )
    return graph_pooler_config


def get_heads(
    class_dict: dict,
    embedding_dim: int = 256,
    head_dropout: float = 0.1,
):
    from omnicons.configs.HeadConfigs import SiameseGraphClsTaskHeadConfig

    heads = {}
    for label_name in labels:
        name = f"bgc_bgc_tanimoto_{label_name}"
        heads[name] = SiameseGraphClsTaskHeadConfig(
            hidden_size=embedding_dim,
            hidden_dropout_prob=head_dropout,
            num_labels=len(class_dict[label_name]),
            class_weight=None,
            analyze_inputs=[("a", "b")],
        )
    return heads


def get_model(
    node_embedding_dim: int = 256,
    node_encoder_dropout: float = 0.1,
    node_encoder_num_layers: int = 1,
    edge_embedding_dim: int = 30,
    edge_encoder_dropout: float = 0.1,
    edge_encoder_num_layers: int = 1,
    edge_type_dropout: float = 0.1,
    edge_type_num_layers: int = 1,
    gat_num_layers: int = 4,
    gat_num_heads: int = 4,
    gat_dropout: float = 0.1,
    transformer_num_layers: int = 4,
    transformer_num_heads: int = 4,
    transformer_attention_dropout: float = 0.1,
    transformer_mlp_dropout: float = 0.1,
    head_dropout: float = 0.1,
    checkpoint_path: Optional[str] = None,
    optimizer: Callable = get_deepspeed_adamw,
):
    # vocab
    node_vocab = get_node_vocab()
    edge_vocab = get_edge_vocab()
    # model setup
    node_encoders = get_node_encoders(
        node_vocab,
        embedding_dim=node_embedding_dim,
        node_encoder_dropout=node_encoder_dropout,
        node_encoder_num_layers=node_encoder_num_layers,
    )
    edge_encoders = get_edge_encoders(
        edge_vocab,
        embedding_dim=edge_embedding_dim,
        edge_encoder_dropout=edge_encoder_dropout,
        edge_encoder_num_layers=edge_encoder_num_layers,
    )
    edge_type_encoder_config = get_edge_type_encoder(
        embedding_dim=edge_embedding_dim,
        edge_type_dropout=edge_type_dropout,
        edge_type_num_layers=edge_type_num_layers,
    )
    gnn_config = get_gnn(
        node_embedding_dim=node_embedding_dim,
        edge_embedding_dim=edge_embedding_dim,
        gat_num_layers=gat_num_layers,
        gat_num_heads=gat_num_heads,
        gat_dropout=gat_dropout,
    )
    transformer_config = get_transformer(
        embedding_dim=node_embedding_dim,
        transformer_num_layers=transformer_num_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_attention_dropout=transformer_attention_dropout,
        transformer_mlp_dropout=transformer_mlp_dropout,
    )
    graph_pooler_config = get_graph_pooler(embedding_dim=node_embedding_dim)
    heads = get_heads(
        class_dict=class_dict,
        embedding_dim=node_embedding_dim,
        head_dropout=head_dropout,
    )
    # Metrics
    train_metrics = ModuleDict()
    val_metrics = ModuleDict()
    for label_name in labels:
        key = f"bgc_bgc_tanimoto_{label_name}___a___b"
        train_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_train",
            num_classes=len(class_dict[label_name]),
            task="multiclass",
        )
        val_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_val",
            num_classes=len(class_dict[label_name]),
            task="multiclass",
        )
    # Instantiate a PyTorch Lightning Module
    model = HeteroGraphModelForMultiTaskLightning(
        node_encoders=node_encoders,
        edge_encoders=edge_encoders,
        edge_type_encoder_config=edge_type_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        graph_pooler_config=graph_pooler_config,
        heads=heads,
        optimizer_fn=optimizer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        inputs=["a", "b"],
        edge_types=edge_types,
    )
    # load checkpoint
    if checkpoint_path != None:
        states = torch.load(checkpoint_path)
        model.load_state_dict(states["state_dict"], strict=False)
    return model
