import argparse
import os

import torch

from omnicons import datdir
from omnicons.models.Compilers import compile_with_torchscript
from training.ibis_bgc.SiameseClassificationTraining import models


def compile_model(torchscript_dir, pytorch_checkpoint_fp):
    if not os.path.exists(torchscript_dir):
        os.makedirs(torchscript_dir)

    if not os.path.exists(f"{torchscript_dir}/node_encoders"):
        os.makedirs(f"{torchscript_dir}/node_encoders")

    if not os.path.exists(f"{torchscript_dir}/edge_encoders"):
        os.makedirs(f"{torchscript_dir}/edge_encoders")

    # load weights
    model = models.get_model()
    states = torch.load(pytorch_checkpoint_fp)
    model.load_state_dict(states["state_dict"], strict=False)

    for node_type, node_encoder in model.model.node_encoders.items():
        compile_with_torchscript(
            model=node_encoder,
            model_fp=f"{torchscript_dir}/node_encoders/{node_type}_node_encoder.pt",
        )

    for edge_type, edge_encoder in model.model.edge_encoders.items():
        compile_with_torchscript(
            model=edge_encoder,
            model_fp=f"{torchscript_dir}/edge_encoders/{edge_type}_edge_encoder.pt",
        )

    compile_with_torchscript(
        model=model.model.edge_type_encoder,
        model_fp=f"{torchscript_dir}/edge_type_encoder.pt",
    )

    compile_with_torchscript(
        model=model.model.gnn, model_fp=f"{torchscript_dir}/gnn.pt"
    )

    compile_with_torchscript(
        model=model.model.transformer,
        model_fp=f"{torchscript_dir}/transformer.pt",
    )

    compile_with_torchscript(
        model=model.model.graph_pooler,
        model_fp=f"{torchscript_dir}/graph_pooler.pt",
    )


parser = argparse.ArgumentParser(
    description="Convert BGC embedding model to pytorch checkpoint."
)
parser.add_argument(
    "-torchscript_dir",
    help="Directory to save torchscript models",
    default=f"{datdir}/siamese_classification_training/torchscript",
)
parser.add_argument(
    "-pytorch_checkpoint_fp",
    help="Pytorch checkpoint file path",
    default=f"{datdir}/siamese_classification_training/checkpoints/last.pt",
)

if __name__ == "__main__":
    args = parser.parse_args()
    compile_model(
        torchscript_dir=args.torchscript_dir,
        pytorch_checkpoint_fp=args.pytorch_checkpoint_fp,
    )
