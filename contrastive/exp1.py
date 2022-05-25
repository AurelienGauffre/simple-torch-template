# python3 exp1.py --wandb --batch_size=128 --group=test3
# oarsub -l /host=1/gpu=4,walltime=24:0:0 '/home/polaris/gauffrea/contrastive/contrastive/config1.sh'

import torch

import pytorch_lightning as pl
import pathlib
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models import ResnetClassique, LinearEvaluation, SwavClassique, MTLSwavSup
from datamodules import Cifar10DataModuleSwaV, Cifar10DataModuleSup, ImagenetteDataModuleSwaV, ImagenetteDataModuleSup, \
    ImagenetteDataModuleSwavSup
from utils import save, load

import argparse

EPOCHS_BASELINE = 200
SWAV_EPOCHS = 600
SWAV_MTL_EPOCHS = 250
EPOCHS_LE_FT = 100
SAVE_EVERY_N_EPOCHS = 200
PROTOTYPES = 512
RANDAUGMENT = True
STRATEGY = 'ddp'

# EPOCHS_BASELINE = 1
# SWAV_EPOCHS = 2
# EPOCHS_LE_FT = 1
# SAVE_EVERY_N_EPOCHS = 1
# SWAV_MTL_EPOCHS = 1

parser = argparse.ArgumentParser(description='Parser of parameters.')
parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
parser.add_argument('--wandb', action='store_true', help='using wandb')
parser.add_argument('--group', type=str, help='group name in wandb', default='Swav-LE-FT')
parser.add_argument('--exp_name', type=str, help='exp name', default='Swav-LE-FT-imagewoof')
parser.add_argument('--dataset', type=str, help='name of dataset', default='imagewoof320px')

params = parser.parse_args()
params.PROTOTYPES = PROTOTYPES
stem = pathlib.Path(__file__).stem if params.exp_name is None else params.exp_name  # default name is the file name
params.root_dir = pathlib.Path(__file__).parent.resolve() / 'checkpoint' / stem

n_gpus = torch.cuda.device_count()
print(f'Training with a batch size of {params.batch_size}')
if STRATEGY == 'ddp':
    params.batch_size = int(params.batch_size / n_gpus)  # todo add warning if not divisble
    params.n_gpus = n_gpus
    params.strategy = STRATEGY
    print(f'DDP Strategy with {n_gpus} GPU: effective batch size on each GPU is {params.batch_size}.')

if __name__ == "__main__":

    # ####################################
    # # Baseline with a resnet :
    # ####################################
    wandb_logger = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                               group=params.group, name='Resnet Baseline RandAugment') if params.wandb else None

    model = ResnetClassique(params, EPOCHS_BASELINE)
    dm_sup = ImagenetteDataModuleSup(params, randaugment=RANDAUGMENT)
    trainer = pl.Trainer(max_epochs=EPOCHS_BASELINE, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True,
                         logger=wandb_logger,
                         default_root_dir=params.root_dir)

    trainer.fit(model=model, datamodule=dm_sup)
    wandb.finish()
    ###################################
    # Baseline with a pretrained-resnet :
    ###################################
    wandb_logger = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                               group=params.group, name='Resnet Pretrained Baseline') if params.wandb else None

    model = ResnetClassique(params, EPOCHS_BASELINE, pretrained=True)
    dm_sup = ImagenetteDataModuleSup(params)
    trainer = pl.Trainer(max_epochs=EPOCHS_BASELINE, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True,
                         logger=wandb_logger,
                         default_root_dir=params.root_dir)

    trainer.fit(model=model, datamodule=dm_sup)
    wandb.finish()

    ###################################
    # SWAV training
    ###################################
    wandb_logger = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                               group=params.group, name='SwaV_pretraining') if params.wandb else None

    model = SwavClassique(params)
    dm_SwaV = ImagenetteDataModuleSwaV(params)
    checkpoint_callback = ModelCheckpoint(dirpath=params.root_dir, filename='SwaV-{epoch}-{train_loss_swav:.2f}',
                                          every_n_epochs=SAVE_EVERY_N_EPOCHS, save_top_k=20, monitor="train_loss_swav")
    trainer = pl.Trainer(max_epochs=SWAV_EPOCHS, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True, logger=wandb_logger,
                         default_root_dir=params.root_dir, callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=dm_SwaV)
    wandb.finish()

    ###################################
    # SWAV Evaluation
    ###################################
    checkpoint_list = list(params.root_dir.glob('SwaV*.ckpt'))
    print(f'Finetuning on :{checkpoint_list}')
    for ckpt in checkpoint_list:
        dm_sup = ImagenetteDataModuleSup(params)
        print(ckpt)
        wandb_logger_FT = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                                      group=params.group,
                                      name=f"FT-SwaV-{ckpt.stem.split('-')[1]}") if params.wandb else None
        # FT
        model_loaded = SwavClassique.load_from_checkpoint(ckpt, params=params)
        modelFT = LinearEvaluation(params, EPOCHS_LE_FT, model_loaded.backbone)
        trainerFT = pl.Trainer(max_epochs=EPOCHS_LE_FT, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True,
                               logger=wandb_logger_FT,
                               default_root_dir=params.root_dir)
        trainerFT.fit(modelFT, dm_sup)
        wandb.finish()

        # FT + randaugment
        dm_sup = ImagenetteDataModuleSup(params)
        print(ckpt)
        wandb_logger_FT = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                                      group=params.group,
                                      name=f"FT+randaugment-SwaV-{ckpt.stem.split('-')[1]}") if params.wandb else None
        model_loaded = SwavClassique.load_from_checkpoint(ckpt, params=params)
        modelFT = LinearEvaluation(params, EPOCHS_LE_FT, model_loaded.backbone)
        trainerFT = pl.Trainer(max_epochs=EPOCHS_LE_FT, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True,
                               logger=wandb_logger_FT,
                               default_root_dir=params.root_dir)
        trainerFT.fit(modelFT, dm_sup)
        wandb.finish()

        # LE
        dm_sup = ImagenetteDataModuleSup(params)
        print(ckpt)
        wandb_logger_LE = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                                      group=params.group,
                                      name=f"LE-SwaV-{ckpt.stem.split('-')[1]}") if params.wandb else None

        model_loaded = SwavClassique.load_from_checkpoint(ckpt, params=params)
        modelLE = LinearEvaluation(params, EPOCHS_LE_FT, model_loaded.backbone, freeze=True)
        trainerLE = pl.Trainer(max_epochs=EPOCHS_LE_FT, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True,
                               logger=wandb_logger_LE,
                               default_root_dir=params.root_dir)
        trainerLE.fit(modelLE, dm_sup)
        wandb.finish()

    # MTL Swav_sup
    wandb_logger = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                               group=params.group, name='MTL SwaV/Sup') if params.wandb else None

    model = MTLSwavSup(SWAV_EPOCHS, params)
    dm_MTL = ImagenetteDataModuleSwavSup(params, randaugment=False)
    trainer = pl.Trainer(max_epochs=SWAV_MTL_EPOCHS, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True,
                         logger=wandb_logger,
                         default_root_dir=params.root_dir)

    trainer.fit(model=model, datamodule=dm_MTL)
    wandb.finish()

    # MTL Swav_sup Scheduler
    wandb_logger = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                               group=params.group, name='MTL SwaV/Sup Scheduler') if params.wandb else None

    model = MTLSwavSup(SWAV_EPOCHS, params, scheduler=True)
    dm_MTL = ImagenetteDataModuleSwavSup(params, randaugment=False)
    trainer = pl.Trainer(max_epochs=SWAV_MTL_EPOCHS, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True,
                         logger=wandb_logger,
                         default_root_dir=params.root_dir)

    trainer.fit(model=model, datamodule=dm_MTL)
    wandb.finish()

    # MTL Swav_sup Randaugment
    wandb_logger = WandbLogger(project='contrastive', entity='aureliengauffre', config=params,
                               group=params.group, name='MTL SwaV/Sup+randaugment ') if params.wandb else None

    model = MTLSwavSup(SWAV_EPOCHS, params)
    dm_MTL = ImagenetteDataModuleSwavSup(params, randaugment=True)
    trainer = pl.Trainer(max_epochs=SWAV_MTL_EPOCHS, gpus=n_gpus, strategy=STRATEGY, sync_batchnorm=True,
                         logger=wandb_logger,
                         default_root_dir=params.root_dir)

    trainer.fit(model=model, datamodule=dm_MTL)
    wandb.finish()
