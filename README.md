# Abstractions of computations experiments
## Installation
- Clone the git repository
- Create a virtual environment with Python 3.10 and activate it
- Install the requirements with `pip install -r torch-requirements.txt && pip install -r requirements.txt`
- For CUDA support (only on Linux), run `pip install -r cuda-requirements.txt`
- Inside the git repo, run `pip install -e .` to install the package in editable mode

## Developer guide
This library is designed to be easy to extend. The two main concepts are *tasks* and
*detectors*.
- A (mechanistic anomaly detection) *task* consists of a *reference dataset*,
  an *anomalous dataset*, and a *model*.
- A *detector* can be trained on the reference dataset and the model. It then needs
  to distinguish whether some sample is from (the test split of) the reference dataset
  or from the anomalous dataset.

This library provides interfaces for tasks and detectors, as well as scripts for
training and evaluating detectors. This means that any new task or detector you
implement will be easy to test against all other detectors/tasks.
