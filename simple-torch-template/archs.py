import torchvision.models as models
import torch
import torch.nn as nn

def get_arch(params):
    if params['arch'] == 'resnet18':
        model = models.resnet18(pretrained=params.get('pretrained', params.pretrained))
    elif params['arch'] == 'resnet50':
        model = models.resnet50(pretrained=params.get('pretrained', params.pretrained))
    else:
        raise ValueError(f"Unsupported architecture: {params['arch']}")

    # Modify the model if necessary (e.g., changing the number of output classes)
    num_classes = params.get('num_classes', 1000)  # Default to 1000 for ImageNet
    if num_classes not in params :
        num_classes = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}[params['dataset']]
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if params['dataset'] in ['cifar10', 'cifar100']:
        print('Modifying model for CIFAR')
        model = modify_resnet_for_cifar10(model)
    return model


def modify_resnet_for_cifar10(model):
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model