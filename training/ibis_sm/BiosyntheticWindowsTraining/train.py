import argparse
import os
from multiprocessing import freeze_support
from typing import Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from omnicons import datdir
from training.ibis_sm.BiosyntheticWindowsTraining.data_modules import (
    BiosyntheticWindowMultiDataModule,
)
from training.ibis_sm.BiosyntheticWindowsTraining.models import get_model


def train(
    checkpoint_dir: str = f"{datdir}/biosyn_windows/checkpoints",
    config_dir: str = f"{datdir}/biosyn_windows/configs",
    weights_fp: str = os.path.join(
        datdir, "biosyn_windows", "biosyn_windows_weights.pkl"
    ),
    pretrained_checkpoint_path: Optional[str] = None,
    checkpoint_name="ibis-sm-{epoch:02d}-{val_loss:.2f}",
    logger_entity="magarvey",
    logger_name="ibis_sm",
    logger_project="ibis",
    trainer_strategy="deepspeed_stage_2_offload",
):
    # Checkpoint Setup
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        save_top_k=3,
        save_last=True,
        mode="min",
        every_n_train_steps=1000,
    )
    # data module
    dm = BiosyntheticWindowMultiDataModule(
        labels_to_consider=("clust_chemotype", "boundary_cat"),
        batch_size=75,
        num_workers=0,
        weights_fp=os.path.join(
            datdir, "biosyn_windows", "biosyn_windows_weights.pkl"
        ),
    )
    dm.setup()
    # model setup
    model = get_model(
        dm=dm,
        config_dir=config_dir,
        weights_fp=weights_fp,
        checkpoint_path=pretrained_checkpoint_path,
    )
    # wandb logger
    wandb_logger = WandbLogger(
        entity=logger_entity,
        name=logger_name,
        project=logger_project,
    )
    # Setup Trainer
    trainer = Trainer(
        max_epochs=10000,
        callbacks=[checkpoint_callback],
        strategy=trainer_strategy,
        precision="16-mixed",
        logger=wandb_logger,
        accelerator="gpu",
        devices="auto",
    )
    # Train
    trainer.fit(model, dm)


################################################################################################

parser = argparse.ArgumentParser(
    description="Train the enzyme classification model."
)
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save model checkpoints",
    default=f"{datdir}/biosyn_windows/checkpoints",
)
parser.add_argument(
    "-config_dir",
    help="Directory to save model configs",
    default=f"{datdir}/biosyn_windows/configs",
)
parser.add_argument(
    "-pretrained_checkpoint_path",
    help="Path to pretrained model checkpoint (if you want to continue training)",
    default=None,
)
parser.add_argument(
    "-weights_fp",
    help="Path to weights file",
    default=os.path.join(
        datdir, "biosyn_windows", "biosyn_windows_weights.pkl"
    ),
)
parser.add_argument(
    "-checkpoint_name",
    help="checkpoint name for wandb",
    default="ibis-sm-{epoch:02d}-{val_loss:.2f}",
)
parser.add_argument(
    "-logger_entity",
    help="wandb entity",
    default="user",
)

if __name__ == "__main__":
    args = parser.parse_args()
    freeze_support()
    train(
        checkpoint_dir=args.checkpoint_dir,
        config_dir=args.config_dir,
        weights_fp=args.weights_fp,
        pretrained_checkpoint_path=args.pretrained_checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        logger_entity=args.logger_entity,
    )
