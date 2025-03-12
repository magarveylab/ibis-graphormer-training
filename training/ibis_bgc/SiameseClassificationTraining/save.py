import argparse

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from omnicons import datdir


def save(
    checkpoint_dir: str = f"{datdir}/training/siamese_classification_training/checkpoints",
):
    convert_zero_checkpoint_to_fp32_state_dict(
        f"{checkpoint_dir}/last.ckpt/", f"{checkpoint_dir}/last.pt"
    )


parser = argparse.ArgumentParser(
    description="Convert BGC embedding model to pytorch checkpoint."
)
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save model checkpoints",
    default=f"{datdir}/training/siamese_classification_training/checkpoints",
)


if __name__ == "__main__":
    args = parser.parse_args()
    save(checkpoint_dir=args.checkpoint_dir)
