import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models import ResnetClassique,
import argparse

parser = argparse.ArgumentParser(description='Parser of parameters.')
parser.add_argument('--bs', type=int, help='batch_size', default=128)
parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
parser.add_argument('--wandb', action='store_true', help='using wandb')
parser.add_argument('--group', type=str, help='group name in wandb', default='contrastive')
parser.add_argument('--run_name', type=str, help='group name in wandb', default=None)
params = parser.parse_args()


if __name__ == "__main__":
    gpus = torch.cuda.device_count()
    wandb_logger = WandbLogger(project='NAS-SSL-MTL', entity='aureliengauffre', config=params,
                               group=params.group, name=params.run_name) if params.wandb else None

    model = ResnetClassique()
    cifar10_dm = Cifar10DataModuleSup()

    # model= SwaV()
    # cifar10_dm = Cifar10DataModuleSwaV()


    trainer = pl.Trainer(
        max_epochs=params.epochs,
        gpus=gpus,
        strategy='ddp',
        sync_batchnorm=True,
        logger=wandb_logger
    )

    trainer.fit(model=model, datamodule=cifar10_dm)


    model2 = LinearEvaluation(model.backbone)
    trainerLinear = pl.Trainer(
        max_epochs=params.epochs,
        gpus=gpus,
        strategy='ddp',
        sync_batchnorm=True,
        logger=wandb_logger
    )
    trainerLinear.fit(model=model2,datamodule=cifar10_dm)