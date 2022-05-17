import torch

import pytorch_lightning as pl
import pathlib
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models import ResnetClassique, LinearEvaluation, SwavClassique
from datamodules import Cifar10DataModuleSwaV, Cifar10DataModuleSup,ImagenetteDataModuleSwaV, ImagenetteDataModuleSup
from utils import save,load

import argparse

SWAV_EPOCHS = 10
SAVE_EVERY_N_EPOCHS = 2

parser = argparse.ArgumentParser(description='Parser of parameters.')
parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
parser.add_argument('--wandb', action='store_true', help='using wandb')
parser.add_argument('--group', type=str, help='group name in wandb', default='test1')
parser.add_argument('--run_name', type=str, help='group name in wandb', default=None)
parser.add_argument('--dataset', type=str, help='name of dataset', default='minimagenette')

params = parser.parse_args()
stem = pathlib.Path(__file__).stem if params.run_name is None else params.run_name #default name is the file name
params.root_dir = pathlib.Path(__file__).parent.resolve() / 'checkpoint' / stem

if __name__ == "__main__":
    gpus = torch.cuda.device_count()
    wandb_logger = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                               group=params.group, name='SwaV_pretraining') if params.wandb else None

    model = SwavClassique(params)
    dm_SwaV = ImagenetteDataModuleSwaV(params)
    checkpoint_callback=ModelCheckpoint(dirpath=params.root_dir,filename='SwaV-{epoch}-{train_loss:.2f}',every_n_epochs=SAVE_EVERY_N_EPOCHS,save_top_k=20,monitor="train_loss")
    trainer = pl.Trainer(max_epochs=SWAV_EPOCHS, gpus=gpus, strategy='ddp', sync_batchnorm=True, logger=wandb_logger,
                         default_root_dir=params.root_dir,callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=dm_SwaV)
    checkpoint_list = list(params.root_dir.glob('SwaV-*.ckpt'))
    print(f'Finetuning on :{checkpoint_list}')
    wandb.finish()
    for ckpt in checkpoint_list:
        dm_sup = ImagenetteDataModuleSup(params)
        print(ckpt)
        wandb_logger_FT = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                                   group=params.group, name=f"FT-SwaV-{ckpt.stem.split('-')[1]}")

        model_loaded = SwavClassique.load_from_checkpoint(ckpt,params=params)
        modelFT = LinearEvaluation(params,model_loaded.backbone)
        trainerFT = pl.Trainer(max_epochs=params.epochs, gpus=gpus, strategy='ddp', sync_batchnorm=True,
                             logger=wandb_logger_FT,
                             default_root_dir=params.root_dir)
        trainerFT.fit(modelFT,dm_sup)
        wandb.finish()

        dm_sup = ImagenetteDataModuleSup(params)
        print(ckpt)

        wandb_logger_LE = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                                      group=params.group, name=f"LE-SwaV-{ckpt.stem.split('-')[1]}")

        model_loaded = SwavClassique.load_from_checkpoint(ckpt, params=params)
        modelLE = LinearEvaluation(params, model_loaded.backbone)
        trainerLE = pl.Trainer(max_epochs=params.epochs, gpus=gpus, strategy='ddp', sync_batchnorm=True,
                               logger=wandb_logger_LE,
                               default_root_dir=params.root_dir)
        trainerLE.fit(modelLE, dm_sup)
        wandb.finish()





    # trainerLinear = pl.Trainer(max_epochs=params.epochs, gpus=gpus, strategy='ddp', sync_batchnorm=True,
    #                            logger=wandb_logger,default_root_dir=params.root_dir)
    #
    # trainerLinear.fit(model=model2, datamodule=cifar10_dm)
