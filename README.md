# Abstractions of computations experiments
## Installation
- Clone the git repository
- Create a virtual environment with Python 3.11 and activate it
- Install the requirements with `pip install -r requirements.txt`
- Inside the git repo, run `pip install -e .` to install the package in editable mode

## Running the experiments
- Run `python -m abstractions.train_mnist` to train a model on MNIST (**currently broken**)
- Run `python -m abstractions.train_abstraction model_path=models/mnist wandb=false`
  to train the abstraction.

## Setting up wandb
This isn't strictly necessary but will be useful later on. Haven't set up a shared
project yet though, so for now just use `wandb=false` when calling training scripts.