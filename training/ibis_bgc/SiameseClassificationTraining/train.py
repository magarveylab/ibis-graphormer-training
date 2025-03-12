import argparse
import os
from multiprocessing import freeze_support

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from omnicons import datdir
from training.ibis_bgc.SiameseClassificationTraining.data_modules import (
    SiameseClsDataModule,
)
from training.ibis_bgc.SiameseClassificationTraining.models import get_model


def train(
    checkpoint_dir: str = f"{datdir}/siamese_classification_training/checkpoints",
    checkpoint_name="ibis-sm-{epoch:02d}-{val_loss:.2f}",
    logger_entity="magarvey",
    logger_name="ibis_bgc",
    logger_project="ibis",
    trainer_strategy="deepspeed_stage_3_offload",
):
    dm = SiameseClsDataModule()
    dm.setup()
    model = get_model()
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
        entity=logger_entity,
        name=logger_name,
        project=logger_project,
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
    trainer.fit(model, dm)


parser = argparse.ArgumentParser(description="Train the BGC comparison model.")
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save model checkpoints",
    default=f"{datdir}/siamese_classification_training/checkpoints",
)
parser.add_argument(
    "-checkpoint_name",
    help="checkpoint name for wandb",
    default="ibis-bgc-{epoch:02d}-{val_loss:.2f}",
)
parser.add_argument(
    "-logger_entity",
    help="wandb entity",
    default="user",
)
parser.add_argument(
    "-logger_name",
    help="wandb entity",
    default="ibis_bgc",
)

if __name__ == "__main__":
    args = parser.parse_args()
    freeze_support()
    train(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        logger_entity=args.logger_entity,
        logger_name=args.logger_name,
    )
