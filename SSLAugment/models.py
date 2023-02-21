""" How to create a model :
- add params, datasetmodule, collate_fn=None  attributes
- add the lightning logging """

import torch
import torchvision
import copy

import pytorch_lightning as pl
import torch.nn as nn

from torchmetrics.functional import accuracy

import lightly
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum, batch_shuffle, batch_unshuffle
from lightly.models.modules import SimSiamProjectionHead
from lightly.models.modules import SimSiamPredictionHead
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.loss import NTXentLoss
from lightly.loss import NegativeCosineSimilarity
from lightly.data import SimCLRCollateFunction

from utils import get_reset_backbone


class LinearEvaluation(pl.LightningModule):
    "Model to train a network in a supervised way from a backbone. Freeze = False allows to finetune the backbone."

    def __init__(self, params, datasetmodule, backbone, freeze=True):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone
        self.max_epochs = params.max_epochs_sup
        self.nb_classes = datasetmodule.nb_classes

        # create a linear layer for the downstream classification model :

        self.projection_head = nn.Linear(512, self.nb_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.freeze = freeze
        if freeze:
            deactivate_requires_grad(self.backbone)

    def forward(self, x):

        z = self.backbone(x).flatten(start_dim=1)
        y_hat = self.projection_head(z)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss_sup', loss, on_step=False,
                 on_epoch=True)
        self.log('train_accuracy', acc, on_step=False,
                 on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True)
        self.log('val_accuracy', acc, on_step=False,
                 on_epoch=True)
        return loss

    def configure_optimizers(self):
        if self.freeze:
            # optim = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=10 ** (-6))
            optim = torch.optim.SGD(self.parameters(), lr=30.)
        else:
            optim = torch.optim.SGD(self.parameters(), lr=0.1,
                                    momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
        # return [optim], [{"scheduler": scheduler, "interval": "epoch"}] A ESSAYER CE TRUC QUE J'AVAIS FAIT


class SimCLRModel(pl.LightningModule):
    """Models have to implement a collate function attributes,
    which will be used in the dataloader when initializing the datamodules."""

    def __init__(self, params, datasetmodule, collate_fn=None):
        super().__init__()

        self.collate_fn = SimCLRCollateFunction(input_size=datasetmodule.input_size,
                                                gaussian_blur=0.,
                                                ) if collate_fn is None else collate_fn
        # self.collate_fn = SimCLRCollateFunction(
        #     input_size=datasetmodule.input_size,
        #     vf_prob=0.5,
        #     rr_prob=0.5
        # ) if collate_fn is None else collate_fn

        # create a ResNet backbone and remove the classification head
        self.backbone = get_reset_backbone(datasetmodule)

        # hidden_dim = resnet.fc.in_features
        # self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        # Le premier 512 correspond a la taille du hidden space a la fin de resnet avant prejeter sur les 1000 classes (cf ci-dessus)
        self.projection_head = SimCLRProjectionHead(512, 512, 128)
        self.criterion = NTXentLoss()
        self.max_epochs = params.max_epochs_ssl

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )
        return [optim], [scheduler]


class SimSiamModel(pl.LightningModule):
    # ADD SCHEDULER !
    def __init__(self, params, datasetmodule, collate_fn=None):
        super().__init__()
        self.collate_fn = lightly.data.SimCLRCollateFunction(input_size=datasetmodule.input_size,
                                                             gaussian_blur=0.,
                                                             ) if collate_fn is None else collate_fn

        self.backbone = get_reset_backbone(datasetmodule)  # todo make cifar10 parametrization automatic
        # resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=1)
        # self.backbone = nn.Sequential(
        #     *list(resnet.children())[:-1],
        #     nn.AdaptiveAvgPool2d(1),  #le resnetgenerator de lightly ne mets pas d'adaptive pooling je crois d'ou ca
        # )

        # Le premier 512 correspond a la taille du hidden space a la fin de resnet avant prejeter sur les 1000 classes
        self.projection_head = SimSiamProjectionHead(512, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        self.criterion = NegativeCosineSimilarity()
        self.max_epochs = params.max_epochs_ssl
    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
                self.parameters(),
                lr=6e-2, # no lr-scaling, results in better training stability
                momentum=0.9,
                weight_decay=5e-4
            )
        # SimSiam seems to perform better without scheduler on Imagenette/Cifar on 100 epochs !
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return optim


    # def configure_optimizers(self):
    #     optim = torch.optim.SGD(
    #         self.parameters(),
    #         lr=6e-2, # no lr-scaling, results in better training stability
    #         momentum=0.9,
    #         weight_decay=5e-4
    #     )
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
    #     return [optim], [scheduler]


class MocoModel(pl.LightningModule):
    def __init__(self, params, datasetmodule, collate_fn=None):
        super().__init__()
        self.max_epochs = params.max_epochs_ssl
        self.collate_fn = lightly.data.SimCLRCollateFunction(input_size=datasetmodule.input_size,
                                                             gaussian_blur=0.,
                                                             ) if collate_fn is None else collate_fn
        memory_bank_size = 4096

        # create a ResNet backbone and remove the classification head
        # resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=1)
        # self.backbone = nn.Sequential(
        #     *list(resnet.children())[:-1],
        #     nn.AdaptiveAvgPool2d(1),  #le resnetgenerator de lightly ne mets pas d'adaptive pooling je crois d'ou ca
        # )
        self.backbone = get_reset_backbone(datasetmodule)  # todo make cifar10 parametrization automatic

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(
            self.projection_head, self.projection_head_momentum, 0.99
        )

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )
        return [optim], [scheduler]
