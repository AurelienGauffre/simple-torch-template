import torch
import torchvision
import torch.nn as nn

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasetmodules import CifarDatasetModule, CifarMiniDatasetModule


def get_embeddings(model, dataloader):
    # We first generale the embeddings
    model.eval()
    embeddings = []
    labels = []
    filenames = []

    with torch.no_grad():
        for img, label, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            labels.append(label)
            filenames.extend(fnames)

    embeddings = normalize(torch.cat(embeddings, 0))
    labels = torch.cat(labels, 0)

    return embeddings, labels, filenames


def knn(k, model, dm):
    # Get training embeddings and label
    train_embeddings, train_labels, train_filenames = get_embeddings(model, dm.train_dataloader())
    # Get val embeddings and label
    val_embeddings, val_labels, val_filenames = get_embeddings(model, dm.val_dataloader())

    # knn training and prediction
    classif = KNeighborsClassifier(n_neighbors=5)
    classif.fit(train_embeddings, train_labels)
    y_predict = classif.predict(val_embeddings)

    return y_predict, val_labels


def save_model(model, path):
    state_dict = {
        'parameters': model.state_dict()
    }
    torch.save(state_dict, path)


def load_model(model, file):
    FILE_EXTENSION = ".ckpt"
    # usage :
    # resnet18= torchvision.models.resnet18()
    # backbone = nn.Sequential(*list(resnet18_new.children())[:-1])
    # load_model(backbone,path)
    path = f"{file}{FILE_EXTENSION}"
    ckpt = torch.load(path)

    model.load_state_dict(ckpt['parameters'])


def init_logger(params, run_name):
    lr_monitor = LearningRateMonitor(logging_interval='epoch') if params.wandb else None
    wandb_logger = WandbLogger(project='SSLAugment', entity='aureliengauffre', config=params,
                               group=params.group, name=run_name, save_dir=params.root_dir,
                               offline=not params.wandb) if params.wandb else None

    # name = f"Test{ckpt.stem.split('-')[1]}"
    # wandb.run.name = 'Run 1'
    return wandb_logger, lr_monitor


def get_reset_backbone(datasetmodule, pretrained=False):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    if type(datasetmodule) in [CifarDatasetModule, CifarMiniDatasetModule]:
        # Modification for cifar10 of resnet
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        resnet.maxpool = nn.Identity()
    return nn.Sequential(*list(resnet.children())[:-1])
