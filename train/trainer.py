import copy
import random
from collections import OrderedDict
import argparse
import math
import time
import warnings

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hypernetworks.hypernetworks import HyperNetwork
from hypernetworks.utils import estimate_connectivity, compute_nbr_params, entropy, estimate_target_sparsity, get_z_interp


class LightningClassifierTask(LightningModule):
    def __init__(self, model, batch_size, latent_size,
                 learning_rate=0.001, monitor=None, mode=None, patience=None, use_sgd=False, lr_reduce=False, use_optim="adam", lr_reduce_factor=0.1, **models_params):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.use_sgd = use_sgd
        self.use_optim = use_optim
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.lr_reduce = lr_reduce
        self.lr_reduce_factor = lr_reduce_factor
        self.metrics_estimated = False
        self.advanced_metrics = False

    def forward(self, x: torch.Tensor, task=None):
        return self.model(x, task)

    def training_step(self, batch, batch_idx):
        x, y, classes, task = batch
        y_hat = self.model(x, task)
        loss = self.loss_function(y_hat, y)
        self.log("Train Loss", loss.detach())
        self.log("Wall time", time.time())
        return loss

    def configure_optimizers(self):
        if self.use_sgd:
            warnings.warn("Deprecated: use 'use_optim' optim")
            self.use_optim = "sgd"
        if self.use_optim == "adam":
            optimizer = optim.Adam
        elif self.use_optim == "radam":
            optimizer = optim.RAdam
        elif self.use_optim == "sgd":
            optimizer = optim.SGD
        else:
            raise ValueError("Invalid optimizer")
        optimizer = optimizer(self.parameters(), lr=self.learning_rate)
        optimizers = {"optimizer": optimizer}
        if self.lr_reduce:
            optimizers["lr_scheduler"] = {
                    "scheduler": ReduceLROnPlateau(optimizer, patience=self.patience//2, factor=self.lr_reduce_factor, mode=self.mode),
                    "monitor": self.monitor,
                }
        return optimizers
    
    def compute_advanced_metrics(self, batch, batch_idx, prefix=""):
        x, y, classes, task = batch
        y_hat = self.model.forward_average_z_model(x)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        return {f'Mean z {prefix} Acc {task.item()}': acc, f'Mean z {prefix} Loss {task.item()}': loss, f'Mean z {prefix} Acc': acc, f'Mean z {prefix} Loss': loss}

    def test_step(self, batch, batch_idx):
        x, y, classes, task = batch
        y_hat = self.model(x, task)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        self.log_dict({f'Test Acc {task.item()}': test_acc})
        nbr_params = compute_nbr_params(self.model)
        entropy_estimate = entropy(y_hat)
        if self.advanced_metrics:
            self.log_dict(self.compute_advanced_metrics(batch, batch_idx, prefix="Test"))
            for i, z in enumerate(get_z_interp(self.model.task_encoder.weight[:, 0], self.model.task_encoder.weight[:, 1])):
                y_hat = self.model.forward_z(x, z)
                labels_hat = torch.argmax(y_hat, dim=1)
                test_acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
                self.log(f"Test Acc {task.item()} interp {i}", test_acc)
        if not self.metrics_estimated:
            self.estimate_metrics()
        return self.log_dict({'Test Loss': loss, 'Test Acc': test_acc, "Params": nbr_params, 'Entropy': entropy_estimate.item()})
    
    def estimate_metrics(self) -> None:
        if isinstance(self.model, HyperNetwork):
            try:
                connectivity, cmin, cmax, _, _, _ = estimate_connectivity(
                    self.model.core, self.latent_size)
                self.log_dict({"Connectivity": connectivity, "Cmin": cmin, "Cmax": cmax})
            except BaseException as err:
                print(f"Error in connectivity estimation: {err}")
            if hasattr(self.hparams, "target_sparsity") and self.hparams.target_sparsity > 0:
                estimated_sparsity = estimate_target_sparsity(self.model, self.latent_size, type="hnet")

        else:
            if hasattr(self.hparams, "target_sparsity") and self.hparams.target_sparsity > 0:
                estimated_sparsity = estimate_target_sparsity(self.model, None, type="experts")
        if hasattr(self.hparams, "target_sparsity") and self.hparams.target_sparsity > 0:
            self.log_dict({"Estimated target sparsity": estimated_sparsity})

        self.metrics_estimated = True
        return

    def validation_step(self, batch, batch_idx):
        x, y, classes, task = batch
        y_hat = self.model(x, task)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        entropy_estimate = entropy(y_hat)
        if self.advanced_metrics:
            self.log_dict(self.compute_advanced_metrics(batch, batch_idx, prefix="Val"))
        return self.log_dict({'Val Loss': loss, 'Val Acc': val_acc, f"Val Acc {task.item()}": val_acc, 'Entropy': entropy_estimate.item()})
