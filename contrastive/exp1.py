import torch

import pytorch_lightning as pl
import pathlib
import os
from pytorch_lightning.loggers import WandbLogger

from models import ResnetClassique, LinearEvaluation
from datamodules import Cifar10DataModuleSwaV, Cifar10DataModuleSup
from utils import save,load

import argparse

parser = argparse.ArgumentParser(description='Parser of parameters.')
parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
parser.add_argument('--wandb', action='store_true', help='using wandb')
parser.add_argument('--group', type=str, help='group name in wandb', default='test')
parser.add_argument('--run_name', type=str, help='group name in wandb', default=None)
params = parser.parse_args()
stem = pathlib.Path(__file__).stem if params.run_name is None else params.run_name
params.root_dir = pathlib.Path(__file__).parent.resolve() / stem

if __name__ == "__main__":
    gpus = torch.cuda.device_count()
    wandb_logger = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                               group=params.group, name=params.run_name) if params.wandb else None

    model = ResnetClassique(params)
    cifar10_dm = Cifar10DataModuleSup(params)

    # model= SwaV()
    # cifar10_dm = Cifar10DataModuleSwaV()

    trainer = pl.Trainer(max_epochs=params.epochs, gpus=gpus, strategy='ddp', sync_batchnorm=True, logger=wandb_logger,
                         default_root_dir=params.root_dir)

    trainer.fit(model=model, datamodule=cifar10_dm)
    trainer.save_checkpoint('last.ckpt')

    model_loaded = ResnetClassique.load_from_checkpoint('last.ckpt')
    model2 = LinearEvaluation(model_loaded.backbone)


    trainerLinear = pl.Trainer(max_epochs=params.epochs, gpus=gpus, strategy='ddp', sync_batchnorm=True,
                               logger=wandb_logger,default_root_dir=params.root_dir)

    trainerLinear.fit(model=model2, datamodule=cifar10_dm)
