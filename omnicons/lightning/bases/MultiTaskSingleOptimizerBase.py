from abc import ABC
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from torch.nn import Module, ModuleDict
from torch_geometric.data import HeteroData

from omnicons.models.DataStructs import MultiTaskOutput


def extract_batch_size(batch) -> Optional[int]:
    if "a" in batch and isinstance(batch["a"], HeteroData):
        nt = batch["a"].node_types[0]
        return torch.unique(batch["a"][nt]["batch"]).shape[0]
    else:
        return None


class MultiTaskSingleOptimizerBase(pl.LightningModule, ABC):
    """A LightningModule for simultaneous training of multiple classification heads with a single optimizer
    Args:
        model: The pretrained base model (pytorch)
        classifier_heads: ModuleDict of heads with the key of their label name (ex. `labels`)
        optimizer_fn (optional): Optimizer factory function
                (see utils.optimizers.preconfigured)
        train_metrics (optional): A ModuleDict of TorchMetrics to be logged
                during training
        val_metrics (optional): A ModuleDict of TorchMetrics to be logged
                during validation
        ignore_index (optional): The value of the label to be ignored when
                calculating metrics
        learning_rate (optional): The learning rate. This can be
                automatically determined with PyTorch Lightning
    """

    def __init__(
        self,
        model: Module,
        optimizer_fn: Callable = None,
        train_metrics: ModuleDict = ModuleDict(),
        val_metrics: ModuleDict = ModuleDict(),
        learning_rate: float = 1e-5,
    ):
        super().__init__()
        self.model = model
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        # Metrics
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs

    def log_metrics(self, metric_dictionary, labels, logits, **kwargs):
        for metric_name, metric in metric_dictionary.items():
            metric(logits, labels)
            self.log(metric_name, metric, **kwargs)

    def training_step(self, batch, batch_idx):
        batch_size = extract_batch_size(batch)
        outputs = self.step(batch=batch, batch_idx=batch_idx)
        if isinstance(outputs, MultiTaskOutput) == False:
            outputs = MultiTaskOutput.merge(
                tensors=outputs, tensor_names=self.model.tensor_names
            )
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        for label_name, logits in outputs.logits.items():
            if label_name not in self.train_metrics:
                continue
            if logits == None:
                continue
            self.log_metrics(
                metric_dictionary=self.train_metrics[label_name],
                labels=outputs.labels[label_name],
                logits=logits,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
        del outputs.logits
        del outputs.labels
        return {"loss": outputs.loss}

    def validation_step(self, batch, batch_idx):
        batch_size = extract_batch_size(batch)
        outputs = self.step(batch=batch, batch_idx=batch_idx)
        if isinstance(outputs, MultiTaskOutput) == False:
            outputs = MultiTaskOutput.merge(
                tensors=outputs, tensor_names=self.model.tensor_names
            )
        self.log(
            "val_loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        for label_name, logits in outputs.logits.items():
            if label_name not in self.val_metrics:
                continue
            if logits == None:
                continue
            self.log_metrics(
                metric_dictionary=self.val_metrics[label_name],
                labels=outputs.labels[label_name],
                logits=logits,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        del outputs

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(
            model_params=self.parameters(), learning_rate=self.learning_rate
        )
        return optimizer
