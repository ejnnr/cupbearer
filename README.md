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

Alternatively, if you're going to do development work on the library itself:
1. Clone this git repository
2. Create a virtual environment with Python 3.10 and activate it
3. Run `pip install -e .` inside the git repo to install the package in editable mode

### Notes on CUDA
On Linux, installing `cupbearer` will install pytorch with CUDA support even though this isn't necessary
(we're only using pytorch for its dataloader and datasets).  If you want to avoid that, run
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
*before* installing `cupbearer`.

Similarly, you may want to [install](https://github.com/google/jax#installation) `jax`
and `jaxlib` manually before installing `cupbearer` to make sure you get the right
CUDA (or CPU only) version.

## Running scripts
You can run any script with the `-h` flag to see a list of options and possible values.

Let's look at an example, where we use a simple Mahalanobis distance-based detection
methods to detect backdoors. First, we need to train a base model on poisoned data:
```bash
python -m cupbearer.scripts.train_classifier \
       --train_data backdoor
       --train_data.original mnist --train_data.backdoor corner \
       --model mlp \
       --dir.full logs/classifier
```
This will train an MLP on MNIST, where the backdoor trigger is that the top left pixel
is set to white (and the model is trained to classify images with this trigger as zeros).
There are many more options like learning rate etc. you can set, but the defaults should
work well enough.

This script will have saved the final model to `logs/classifier/model/`. We can now train
a Mahalanobis-based detector on this model:
```bash
python -m cupbearer.scripts.train_detector \
       --detector mahalanobis --task backdoor \
       --task.backdoor corner --task.run_path logs/classifier \
       --dir.full logs/detector
```
(The fact that we need to specify the backdoor trigger again is an artifact of how
things are currently implemented and will likely be changed---the detector is only actually
trained on clean images). `train_detector` loads the model from `task.run_path` and
also uses that to figure out which dataset to use (in this case, MNIST again).

Finally, we can evaluate the detector on a mix of clean and poisoned images:
```bash
python -m cupbearer.scripts.eval_detector
       --detector from_run --detector.path logs/detector \
       --task backdoor --task.backdoor corner \
       --task.run_path logs/classifier \
       --dir.full logs/detector
```
Note that we reuse `logs/detector` as the output directory. This will add the evaluation
results to the directory of the detector training run (it won't overwrite anything).

The reason we need to specify some things, such as the task, multiple times is that
these scripts also support much more flexible setups. For example, we could evaluate
the detector on images with a *different* backdoor trigger (as an ablation to check
that these are *not* flagged as anomalous), in which case we might also want
to save the results to a different directory:
```bash
python -m cupbearer.scripts.eval_detector
       --detector from_run --detector.path logs/detector \
       --task backdoor --task.backdoor noise --task.backdoor.std 0.2 \
       --task.run_path logs/classifier \
       --dir.full logs/detector_ablation \
```

In the future, there might be easier ways of getting "reasonable defaults" for some settings,
for now you could just write small wrapper scripts for these kinds of pipelines.

## Whence the name?
Just like a cupbearer tastes wine to avoid poisoning the king, mechanistic anomaly
detection methods taste new inputs to check whether they're poisoned. (Or more generally,
anomalous in terms of how they affect the king ... I mean, model. I admit the analogy
is becoming a bit strained here.)
