# `cupbearer` 🍷
`cupbearer` is a Python library for
[mechanistic anomaly detection](https://www.alignmentforum.org/posts/vwt3wKXWaCvqZyF74/mechanistic-anomaly-detection-and-elk).
Its main purpose is to make it easy to implement either a new mechanistic anomaly
detection *task*, or a new detection *method*, and evaluate it against a suite of existing
tasks or methods. To that end, the library provides:
- A clearly defined interface for tasks and detectors to ensure compatibility
- Scripts and other helpers for training and evaluating detectors that satisfy this interface
  (to reduce the amount of boilerplate code you need to write)
- Implementations of several tasks and detectors as baselines (currently not that many)

Contributions of new tasks or detectors are very welcome!
See the [developer guide](docs/getting_started.md) to get started.

## Installation
The easy way: inside a virtual environment with Python >= 3.10, run
```bash
pip install git+https://github.com/ejnnr/cupbearer.git
```
(You could also `pip install cupbearer`, but note that the library is under heavy
development and the PyPi version will often be outdated.)

Alternatively, if you're going to do development work on the library itself:
1. Clone this git repository
2. Create a virtual environment with Python 3.10 and activate it
3. Run `pip install -e .` inside the git repo to install the package in editable mode

### Notes on Pytorch
Depending on what platform you're on, you may need to install Pytorch separately *before*
installing `cupbearer`, in particular if you want to control CUDA version etc.

## Running scripts
You can run any script with the `-h` flag to see a list of options and possible values.

Let's look at an example, where we use a simple Mahalanobis distance-based detection
method to detect backdoors. First, we need to train a base model on poisoned data:
```bash
python -m cupbearer.scripts.train_classifier \
       --train_data backdoor --train_data.original mnist \
       --train_data.backdoor corner --train_data.backdoor.p_backdoor 0.1 \
       --model mlp --num_epochs 2 \
       --dir.full logs/demo/classifier
```
This will train an MLP on MNIST, where the backdoor trigger is that the top left pixel
is set to white (and the model is trained to classify images with this trigger as zeros).
There are many more options like learning rate etc. you can set, but the defaults should
work well enough.

This script will have saved the final model to `logs/demo/classifier/checkpoints/last.ckpt`.
We can now train a Mahalanobis-based detector on this model:
```bash
python -m cupbearer.scripts.train_detector \
       --detector mahalanobis --task backdoor \
       --task.backdoor corner --task.path logs/demo/classifier \
       --dir.full logs/demo/mahalanobis
```
(The fact that we need to specify the backdoor trigger again is an artifact of how
things are currently implemented and will likely be changed---the detector is only actually
trained on clean images). `train_detector` loads the model from `task.path` and
also uses that to figure out which dataset to use (in this case, MNIST again).

If everything works, this should print out an AUCROC and AP of >0.99.
(This detection task is very easy.) Inside the `logs/demo/mahalanobis` directory,
there should also be a `histogram.pdf` plot that shows the distribution of anomaly
scores. Backdoored images and clean images should be very well separated.

We could also evaluate the detector on images with a *different* backdoor trigger
(as an ablation to check that these are *not* flagged as anomalous). We can use
the `eval_detector` script for this (which is also what `train_detector` uses
internally to produce the evaluation results at the end):
```bash
python -m cupbearer.scripts.eval_detector \
       --detector.path logs/demo/mahalanobis \
       --task backdoor --task.backdoor noise --task.backdoor.std 0.1 \
       --task.path logs/demo/classifier \
       --dir.full logs/demo/mahalanobis_ablation
```
This will still have an AUCROC and AP greater than 0.5 because the noise is
somewhat anomalous in terms of Mahalanobis distance, but it should be closer to 0.5
than before.

## Whence the name?
Just like a cupbearer tastes wine to avoid poisoning the king, mechanistic anomaly
detection methods taste new inputs to check whether they're poisoned. (Or more generally,
anomalous in terms of how they affect the king ... I mean, model. I admit the analogy
is becoming a bit strained here.)
