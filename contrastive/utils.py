import torch
import torchvision
from torch import nn


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    # state_dict = torch.load(model_path)
    # # create new ordererDict that does not contain 'module'
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #   namekey = k[7:] # remove 'module'
    #   new_state_dict[namekey] = v
    # # load params
    model.load_state_dict(torch.load(model_path))


def get_reset_backbone(cifar10=False):
    resnet = torchvision.models.resnet18()
    if cifar10:
        # Modification for cifar10 of resenet
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        resnet.maxpool = nn.Identity()
    return nn.Sequential(*list(resnet.children())[:-1])
