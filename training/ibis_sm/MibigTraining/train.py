import argparse
import os
from multiprocessing import freeze_support
from typing import List

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from omnicons import datdir
from training.ibis_sm.MibigTraining.data_modules import KFoldDataModule
from training.ibis_sm.MibigTraining.models import get_model


def train(
    kfold_iteration: int,
    config_dir: str = f"{datdir}/mibig_training/configs",
    chemotypes_to_consider: List[str] = [
        "NRP",
        "Alkaloid",
        "Polyketide",
        "RiPP",
        "Saccharide",
        "Terpene",
        "Other",
    ],
    suffix_options: List[str] = ["a", "b", "c", "d", "e", "f", "g"],
    pretrained_checkpoint_path: str = os.path.join(
        datdir, "mibig_training", "pretrained_checkpoints", "ibis_sm.pt"
    ),
    checkpoint_dir: str = f"{datdir}/mibig_training/checkpoints",
    checkpoint_name="ibis-sm-mibig-{epoch:02d}-{val_loss:.2f}",
    logger_entity="magarvey",
    logger_name="ibis_sm_mibig",
    logger_project="ibis",
    trainer_strategy="deepspeed_stage_3_offload",
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
    # wandb logger
    wandb_logger = WandbLogger(
        entity=logger_entity, name=logger_name, project=logger_project
    )
    # data module
    dm = KFoldDataModule(
        kfold_iteration=kfold_iteration,
        chemotypes_to_consider=chemotypes_to_consider,
        suffix_options=suffix_options,
    )
    dm.setup()
    # model setup
    model = get_model(
        dm=dm,
        config_dir=config_dir,
        checkpoint_path=pretrained_checkpoint_path,
        chemotypes_to_consider=chemotypes_to_consider,
        suffix_options=suffix_options,
    )
    # Setup Trainer
    trainer = Trainer(
        max_epochs=100000,
        callbacks=[checkpoint_callback],
        strategy=trainer_strategy,
        precision="16-mixed",
        logger=wandb_logger,
        accelerator="gpu",
        devices="auto",
    )
    # Train
    trainer.fit(model, dm)


parser = argparse.ArgumentParser(
    description="Train the BGC boundary/chemotype classification model."
    "on MIBiG data."
)
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save model checkpoints",
    default=f"{datdir}/mibig_training/checkpoints",
)
parser.add_argument(
    "-config_dir",
    help="Directory to save model configs",
    default=f"{datdir}/mibig_training/configs",
)
parser.add_argument(
    "-pretrained_checkpoint_path",
    help="Path to pretrained model checkpoint",
    default=os.path.join(
        datdir, "mibig_training", "pretrained_checkpoints", "ibis_sm.pt"
    ),
)
parser.add_argument(
    "-checkpoint_name",
    help="checkpoint name for wandb",
    default="ibis-sm-mibig-{epoch:02d}-{val_loss:.2f}",
)
parser.add_argument(
    "-logger_entity",
    help="wandb entity",
    default="user",
)
parser.add_argument(
    "-logger_name",
    help="wandb entity",
    default="ibis_sm_mibig",
)
parser.add_argument(
    "-kfold",
    help="kfold split to run, -1 for all",
    required=True,
)

if __name__ == "__main__":
    args = parser.parse_args()
    freeze_support()
    train(
        kfold_iteration=args.kfold,
        checkpoint_dir=args.checkpoint_dir,
        config_dir=args.config_dir,
        weights_fp=args.weights_fp,
        pretrained_checkpoint_path=args.pretrained_checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        logger_entity=args.logger_entity,
    )
