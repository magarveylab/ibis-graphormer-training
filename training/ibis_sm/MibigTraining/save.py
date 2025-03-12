import argparse

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from omnicons import datdir


def save(
    checkpoint_dir: str = f"{datdir}/training/mibig_training/checkpoints",
):
    convert_zero_checkpoint_to_fp32_state_dict(
        f"{checkpoint_dir}/last.ckpt/", f"{checkpoint_dir}/last.pt"
    )


parser = argparse.ArgumentParser(
    description="Convert MIBiG-finetuned BGC identification model to "
    "pytorch checkpoint."
)
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save model checkpoints",
    default=f"{datdir}/training/mibig_training/checkpoints",
)
