import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchmetrics.functional import accuracy
from torchvision.models import resnet18, resnet34, resnet50, vgg16, vgg16_bn
# from archs.resnet import ResNet18, ResNet34, ResNet50, ResNet101
import torch
import torch.nn as nn
import time

# supported arch
dic_arch = {
    'resnet18': resnet18(),
    'resnet18_noBN': resnet18(),
    'resnet34': resnet34(),
    'resnet34_noBN': resnet34(),
    'resnet50': resnet50(),
    'resnet50_noBN': resnet50(),
    'vgg16': vgg16(),
    'vgg16_bn': vgg16_bn()
}


class MultiRegression(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.start_time = None
        self.lr = params.lr
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.img_size = params.img_size
        self.criterion = nn.MSELoss()
        # todo create arch cleanly from a separate file
        self.list_labels = params.list_labels
        if 'Ev0' in params.list_labels:
            self.ev0_index = params.list_labels.index('Ev0')
        self.arch = self.get_arch(params)

        # disabling bachnorm for specific arch
        def disable_batchnorm(m):
            if isinstance(m, nn.BatchNorm2d):
                print(f"Disabling batchnorm for {m}")
                m = Identity()
        if 'noBN' in params.arch:
            self.arch.apply(disable_batchnorm)

    def forward(self, x):
        out = self.arch(x)
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log_dict({"train/loss": loss}, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def evaluate(self, batch, stage=None):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        mae = torch.mean(torch.abs(outputs - targets), dim=0)

        relative_error = 100 * torch.mean(torch.abs(outputs - targets) / (targets + 1e-5), dim=0)  # todofix target

        if stage:
            # print(torch.mean(torch.abs(outputs - targets)))
            logdict = {
                **{f"{stage}/loss": loss, f"gpu memory": round(torch.cuda.max_memory_allocated() / 1024.0 ** 3, 1)},
                **self.get_log_dict(mae, relative_error, stage)}
            self.log_dict(logdict,
                          prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs),
            "interval": "epoch",  # steps or epoch ? depends on the previous line
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        end_time = time.time()
        total_time = (end_time - self.start_time) / 3600
        self.log_dict({"total time": round(total_time, 2), "epoch avg time": round(total_time / self.epochs, 2)},
                      on_epoch=True,
                      on_step=False)

    def get_arch(self, params):
        """
           Configures the architecture of the model based on the provided parameters.
        """

        self.arch = dic_arch[params.arch]
        # modify the backbone to make it suitable for our classification
        if "resnet" in params.arch:
            num_in_features = self.arch.fc.in_features
            self.arch.fc = nn.Linear(num_in_features, params.nb_labels)

            old_conv_weight = self.arch.conv1.weight.data
            self.arch.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.arch.conv1.weight.data = old_conv_weight.mean(dim=1, keepdim=True)
        elif "vgg" in params.arch:
            num_in_features = self.arch.classifier[6].in_features
            self.arch.classifier[6] = torch.nn.Linear(num_in_features, params.nb_labels)

            # Change the first layer
            old_conv_weight = self.arch.features[0].weight.data
            self.arch.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Here we keep the average weights from the three input channels
            self.arch.features[0].weight.data = old_conv_weight.mean(dim=1, keepdim=True)
        else:
            raise ValueError(
                f"Unsupported architecture: {params.arch}. Supported architectures are 'resnet' and 'vgg'.")
        return self.arch

    def get_log_dict(self, mae, relative_error, stage):
        """

        """
        dic = {}
        for i, (label, mae_i) in enumerate(zip(self.list_labels, mae)):
            dic[f'{stage}/MAE_{label}'] = mae[i].item()
            dic[f'{stage}/relative_error_{label}'] = relative_error[i].item()
        if "Ev0" in self.list_labels:
            dic[f'{stage}/MAE_Ev_total'] = torch.mean(mae[self.ev0_index:]).item()
            dic[f'{stage}/relative_error_Ev_total'] = torch.mean(relative_error[self.ev0_index:]).item()
        return dic



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

