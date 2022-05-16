import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class Cifar10DataModuleSup(pl.LightningDataModule):
    def __init__(self, params, data_dir: str = "~/datasets/cifar10" ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = params.batch_size

    def train_dataloader(self):
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
        train_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True),transform=train_transforms)
        return torch.utils.data.DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           )


    def val_dataloader(self):
        val_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
        val_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True),transform=val_transforms)
        return torch.utils.data.DataLoader(val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           )



class Cifar10DataModuleSwaV(pl.LightningDataModule):
    def __init__(self, params, data_dir: str = "~/datasets/cifar10"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = params.batch_size

    def train_dataloader(self):
        train_transforms = None
        train_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True), target_transform=lambda t: 0)
        collate_fn = SwaVCollateFunction()
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )
