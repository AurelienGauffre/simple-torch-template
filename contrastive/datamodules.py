import torch
from torch import nn
import torchvision
import PIL
import os
import pathlib
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
        train_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True, target_transform=lambda t: 0))
        collate_fn = SwaVCollateFunction()
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )


class ImagenetteDataModuleSwaV(pl.LightningDataModule):
    def __init__(self, params, data_dir=pathlib.Path(os.path.expanduser('~/datasets'))):
        super().__init__()
        self.params = params
        self.data_dir = data_dir /  params.dataset


    def train_dataloader(self):
        train_transforms = None

        train_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.ImageFolder(self.data_dir, target_transform=lambda t: 0))
        collate_fn = SwaVCollateFunction()
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.params.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )


class ImagenetteDataModuleSup(pl.LightningDataModule):

    def __init__(self, params, data_dir=pathlib.Path(os.path.expanduser('~/datasets')),randaugment=None):
        super().__init__()
        self.params= params
        self.data_dir = data_dir /  params.dataset
        self.randaugment =randaugment
    def train_dataloader(self):
        RANDAUGMENT=[torchvision.transforms.RandAugment(num_ops=2,magnitude=9) if self.randaugment is not None else []]

        train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, PIL.Image.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        *RANDAUGMENT,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        train_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.ImageFolder(self.data_dir/'train'),transform=train_transforms)
        return torch.utils.data.DataLoader(train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           )

    def val_dataloader(self):
        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224), #TOCHANGE : center crop or directly cutting to 256 ?
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
        val_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.ImageFolder(self.data_dir/'val'),transform=val_transforms)
        return torch.utils.data.DataLoader(val_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           )



class ImagenetteDataModuleSwavSup(pl.LightningDataModule):

    def __init__(self, params, data_dir=pathlib.Path(os.path.expanduser('~/datasets')),randaugment=None):
        super().__init__()
        self.params= params
        self.data_dir = data_dir /  params.dataset
        self.randaugment =randaugment
    def train_dataloader(self):
        # Sup loader
        RANDAUGMENT=[torchvision.transforms.RandAugment(num_ops=2,magnitude=9) if self.randaugment is not None else []]

        train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, PIL.Image.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        *RANDAUGMENT,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        Sup_train_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.ImageFolder(self.data_dir/'train'),transform=train_transforms)
        Sup_loader = torch.utils.data.DataLoader(Sup_train_dataset,
                                    batch_size=self.params.batch_size,
                                    shuffle=True,
                                    num_workers=8,
                                    )

        # Swav loader
        SwaV_train_dataset = LightlyDataset.from_torch_dataset(
            torchvision.datasets.ImageFolder(self.data_dir, target_transform=lambda t: 0))
        collate_fn = SwaVCollateFunction()
        Swav_loader = torch.utils.data.DataLoader(
            SwaV_train_dataset,
            batch_size=self.params.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )

        #Combning both https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html#return-multiple-dataloaders
        loaders = {"Sup": Sup_loader, "SwaV": Swav_loader}
        return loaders

    def val_dataloader(self):
        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224), #TOCHANGE : center crop or directly cutting to 256 ?
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
        val_dataset = LightlyDataset.from_torch_dataset(torchvision.datasets.ImageFolder(self.data_dir/'val'),transform=val_transforms)
        return torch.utils.data.DataLoader(val_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           )

