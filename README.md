# Abstractions of computations experiments
## Installation
- Clone the git repository
- Create a virtual environment with Python 3.10 and activate it
- Install the requirements with `pip install -r torch-requirements.txt && pip install -r requirements.txt`
- For CUDA support (only on Linux), run `pip install -r cuda-requirements.txt`
- Inside the git repo, run `pip install -e .` to install the package in editable mode

## Running the experiments
- Run `python -m abstractions.train_mnist wandb=false` to train a model on MNIST
- Run `python -m abstractions.train_abstraction base_run=/logs/based/some-directory wandb=false`
  to train the abstraction.

## Setting up wandb
This isn't strictly necessary but will be useful later on. Haven't set up a shared
project yet though, so for now just use `wandb=false` when calling training scripts.
