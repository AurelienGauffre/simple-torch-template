# python3 experiment/models.py --wandb --bs=256
# voir le tuto de barlow twin de lightning sur comment modifier l'architecture quand il s'agit de resnet https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/barlow-twins.html
import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SimCLRProjectionHead
from lightly.models.modules import SwaVPrototypes
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import argparse

from utils import get_reset_backbone
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR

from pl_bolts.datamodules import CIFAR10DataModule

import wandb


class SwavClassique(pl.LightningModule):
    def __init__(self,params):
        super().__init__()
        self.params = params
        self.backbone = get_reset_backbone(cifar10=False)
        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=params.PROTOTYPES)

        # enable sinkhorn_gather_distributed to gather features from all gpus
        # while running the sinkhorn algorithm in the loss calculation
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True, sinkhorn_epsilon=0.03)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        crops, _, _ = batch
        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True)  # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#train-epoch-level-operations
        # self.log('train_accuracy', acc)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


class ResnetClassique(pl.LightningModule):
    def __init__(self,params):
        super().__init__()
        self.backbone = get_reset_backbone(cifar10=False)
        self.projection_head = SimCLRProjectionHead(512, 512, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.params=params

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        # x = nn.functional.normalize(x, dim=1, p=2)
        # p = self.prototypes(x)
        return x

    def training_step(self, batch, batch_idx):
        # self.prototypes.normalize()
        # crops, _, _ = batch
        # multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        # high_resolution = multi_crop_features[:2]
        # low_resolution = multi_crop_features[2:]
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True)  # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#train-epoch-level-operations
        self.log('train_accuracy', acc, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, stage=None):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True)
        self.log('val_accuracy', acc, on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.params.epochs)
        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]


class LinearEvaluation(pl.LightningModule):
    """ Model to train a linear classifier on top of a backbone encoder. Use --"""

    def __init__(self, params,backbone, freeze: bool = False):
        super().__init__()
        self.backbone = backbone
        self.params= params
        self.freeze = freeze

        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False

        self.projection_head = nn.Linear(512, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                x = self.backbone(x).flatten(start_dim=1)
        else:
            x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True)  # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#train-epoch-level-operations
        self.log('train_accuracy', acc, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, stage=None):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True)
        self.log('val_accuracy', acc, on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.params.epochs)
        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]




class SwavSup(pl.LightningModule):
    def __init__(self,params):
        super().__init__()
        self.params = params
        self.backbone = get_reset_backbone(cifar10=False)
        self.projection_head_swav = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=params.PROTOTYPES)

        # enable sinkhorn_gather_distributed to gather features from all gpus
        # while running the sinkhorn algorithm in the loss calculation
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True, sinkhorn_epsilon=0.03)
        self.projection_head_sup = nn.Linear(512, 10)
        self.criterion_sup = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        crops, _, _ = batch
        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True)  # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#train-epoch-level-operations
        # self.log('train_accuracy', acc)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim
