import itertools
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
import wandb
from archs import get_arch
from datasets import get_loaders
import os


def run(params):
    wandb.init(project=params.wandb_project, config=params)

    criterion = torch.nn.CrossEntropyLoss()
    model = get_arch(params)
    train_loader, val_loader, test_loader = get_loaders(params)
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_epochs)
    # Add dataloader, optimizer, criterion based on cfg

    for epoch in range(params.num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, 'train', params)
        if epoch % params['eval_every_n_epochs'] == 0:
            val_loss, val_accuracy = train(model, val_loader, criterion, optimizer, 'val', params)
        scheduler.step()

    if test_loader:
        test_loss, test_accuracy = train(model, test_loader, criterion, optimizer, 'test', params)

    wandb.finish()


def train(model, dataloader, criterion, optimizer, phase, params):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    for data, target in dataloader:
        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            output = model(data)
            loss = criterion(output, target)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()
        print("current accuracy: {}".format((predicted == target).sum().item() / params.bs))

    avg_loss = total_loss / len(dataloader.dataset)
    avg_accuracy = total_correct / len(dataloader.dataset)
    print(f"{phase.capitalize()}: Avg. Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    # Log metrics to wandb
    log_dict = {f"{phase}/loss": avg_loss, f"{phase}/acc": avg_accuracy}
    if phase == 'train':
        log_dict[f"{phase}/lr"] = optimizer.param_groups[0]['lr']
    wandb.log(log_dict)

    return avg_loss, avg_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', type=str, default='config.yaml')
    args_command_line = parser.parse_args()
    config_dict = OmegaConf.to_container(
        OmegaConf.load(os.path.join('configs',
                                    args_command_line.config)))  # Convert the yaml file into a classical python dictionary

    # The following loop is used to iterate over all the possible combinations of hyperparameters that are given
    # as lists in the config file
    list_keys = [k for k, v in config_dict.items() if isinstance(v, list)]  # Identify which keys have list values
    combinations = list(itertools.product(*(config_dict[k] for k in list_keys)))
    for values in combinations:
        for k, v in zip(list_keys, values):
            config_dict[k] = v
        config = OmegaConf.create(config_dict)
        print(config)
        run(config)
