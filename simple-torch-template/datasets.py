import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_loaders(params):
    # Define dataset-specific transformations
    if params['dataset'] in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif params['dataset'] == 'imagenet':
        # Example transformations for ImageNet
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Unsupported dataset")

    # Load the appropriate dataset
    if params['dataset'] == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif params['dataset'] == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif params['dataset'] == 'imagenet':
        # Replace with the path to your ImageNet data
        imagenet_path = '/path/to/imagenet'
        train_dataset = datasets.ImageFolder(root=f'{imagenet_path}/train', transform=transform_train)
        test_dataset = datasets.ImageFolder(root=f'{imagenet_path}/val', transform=transform_test)
    else:
        raise ValueError("Unsupported dataset")

    # Creating data indices for training and validation splits
    # Determine the number of samples for the downsampling
    full_train_indices = np.random.permutation(len(train_dataset))

    # Split the full train indices into training and validation subsets
    split = int(np.floor(0.1 * len(full_train_indices)))
    train_indices, val_indices = full_train_indices[split:], full_train_indices[:split]

    # Downsample the training and validation indices
    downsample_train_size = int(np.floor(params.fraction * len(train_indices)))
    downsample_val_size = int(np.floor(params.fraction * len(val_indices)))

    downsampled_train_indices = train_indices[:downsample_train_size]
    downsampled_val_indices = val_indices[:downsample_val_size]

    # Creating subsets of the datasets
    train_subset = Subset(train_dataset, downsampled_train_indices)
    val_subset = Subset(train_dataset, downsampled_val_indices)
    test_subset = Subset(test_dataset,
                         np.random.permutation(len(test_dataset))[:int(np.floor(params.fraction * len(test_dataset)))])

    # Creating data loaders for each subset
    train_loader = DataLoader(train_subset, batch_size=params['bs'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=params['bs'], shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=params['bs'], shuffle=False)

    return train_loader, val_loader, test_loader