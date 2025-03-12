import argparse
import os
from glob import glob

import torch

from omnicons import datdir
from omnicons.configs.HeadConfigs import NodeClsTaskHeadConfig
from omnicons.lightning.GraphModelForMultiTask import (
    GraphModelForMultiTaskLightning,
)
from omnicons.models.Compilers import compile_with_torchscript
from training.ibis_sm.MibigTraining import models as pretrained_model


def compile_model(
    config_dir,
    torchscript_dir,
    pytorch_checkpoint_fp,
):
    if not os.path.exists(torchscript_dir):
        os.makedirs(torchscript_dir)
    # load heads
    heads = {}
    head_fps = [
        x for x in glob(os.path.join(config_dir, "*.json")) if "head" in x
    ]
    for head_fp in head_fps:
        head_name = os.path.basename(head_fp).replace("_head_config.json", "")
        heads[head_name] = NodeClsTaskHeadConfig.load_from(head_fp)
    #  load model
    node_encoder_config = pretrained_model.get_node_encoder(
        config_dir=config_dir,
    )
    gnn_config = pretrained_model.get_gnn(config_dir=config_dir)
    transformer_config = pretrained_model.get_transformer(
        config_dir=config_dir
    )
    model = GraphModelForMultiTaskLightning(
        node_encoder_config=node_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        heads=heads,
    )
    # load weights
    states = torch.load(pytorch_checkpoint_fp)
    model.load_state_dict(states["state_dict"], strict=False)
    # export models (use /home/norman/.conda/envs/omnicons env instead of pyG)
    compile_with_torchscript(
        model=model.model.node_encoder,
        model_fp=f"{torchscript_dir}/node_encoder.pt",
    )
    compile_with_torchscript(
        model=model.model.gnn, model_fp=f"{torchscript_dir}/gnn.pt"
    )
    compile_with_torchscript(
        model=model.model.transformer,
        model_fp=f"{torchscript_dir}/transformer.pt",
    )
    for head_name, head in heads.items():
        head_model = getattr(model.model.heads, head_name, None)
        assert head_model is not None
        compile_with_torchscript(
            model=head_model,
            model_fp=f"{torchscript_dir}/{head_name}.pt",
        )


parser = argparse.ArgumentParser(
    description="Convert BGC identification model to pytorch checkpoint."
)
parser.add_argument(
    "-torchscript_dir",
    help="Directory to save torchscript models",
    default=f"{datdir}/biosyn_windows/torchscript",
)
parser.add_argument(
    "-pytorch_checkpoint_fp",
    help="Pytorch checkpoint file path",
    default=f"{datdir}/biosyn_windows/checkpoints/last.pt",
)
parser.add_argument(
    "-config_dir",
    help="Directory containing model config files",
    default=f"{datdir}/biosyn_windows/configs",
)

if __name__ == "__main__":
    args = parser.parse_args()
    compile_model(
        config_dir=args.config_dir,
        torchscript_dir=args.torchscript_dir,
        pytorch_checkpoint_fp=args.pytorch_checkpoint_fp,
    )
