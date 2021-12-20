import copy
import random
from collections import OrderedDict
import argparse
import math

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hypernetworks.hypernetworks import HyperNetwork
from hypernetworks.utils import estimate_connectivity, compute_nbr_params


class LightningClassifierTask(LightningModule):
    def __init__(self, model, batch_size, latent_size,
                 learning_rate=0.001, monitor=None, patience=None, use_sgd=False):
        super().__init__()
        self.model = model
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.use_sgd = use_sgd
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.monitor = monitor
        self.patience = patience

    def forward(self, x: torch.Tensor, task=None):
        return self.model(x, task)

    def training_step(self, batch, batch_idx):
        x, y, classes, task = batch
        y_hat = self.model(x, task)
        loss = self.loss_function(y_hat, y)
        self.log("Train Loss", loss.detach())
        return loss

    def configure_optimizers(self):
        if self.use_sgd:
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, patience=self.patience//2, factor=0.1),
                    "monitor": self.monitor,
                },
            }
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def test_step(self, batch, batch_idx):
        x, y, classes, task = batch
        y_hat = self.model(x, task)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        self.log_dict({f'Test Acc {task.item()}': test_acc})
        nbr_params = compute_nbr_params(self.model)
        if isinstance(self.model, HyperNetwork):
            connectivity, cmin, cmax, _, _, _ = estimate_connectivity(
                self.model.core, self.latent_size)
            self.log_dict({"Connectivity": connectivity, "Cmin": cmin, "Cmax": cmax})
        return self.log_dict({'Test Loss': loss, 'Test Acc': test_acc, "Params": nbr_params})

    def validation_step(self, batch, batch_idx):
        x, y, classes, task = batch
        y_hat = self.model(x, task)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        return self.log_dict({'Val Loss': loss, 'Val Acc': val_acc, f"Val Acc {task.item()}": val_acc})