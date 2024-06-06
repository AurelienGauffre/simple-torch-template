# ⚡ Simple-Torch-Template

This is a very basic and simple template for PyTorch projects that utilizes PyTorch and Weight and Biases (Wandb).
The goal of this repo is to save a few minutes when creating PyTorch projects from scratch or adapting existing
ones.

# Note

The configuration file is currently managed using the Omegaconf package, which simply allows to transform the YAML 
config file into a python object similar to argparse. Using Hydra, which also uses Omegaconf, would
currently introduce too much complexity since it still create some frictions with other libraries and functionalities 
such as DDP training or W&B sweeps.

# Requirements

* Python >= 3.5
* Pytorch
* Wandb
* Omegaconf

# Features

* Easy hyperparameters management with YAML config files with Omegaconf
* Easy Grid Search experiment using list in YAML file 
* Clean logging of the metrics with W&B

# Todo :
- [ ] Multi GPU with DDP
- [ ] Mix precision support
- [ ] Wandb sweeps
- [ ] Advanced scheduling
- [ ] Clean architecture loading from arch files (add path or create package)
- [ ] Dependency management with virtual env
- [ ] Multiple datasets
- [ ] Add basic/intermediate examples in this repo
  (e.g. contrastive learning with fine-tuning, semi-supervised learning,
  segmentation or NLP tasks)

