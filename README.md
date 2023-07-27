# Abstractions of computations experiments
## Installation
- Clone the git repository
- Create a virtual environment with Python 3.10 and activate it
- Install the requirements with `pip install -r torch-requirements.txt && pip install -r requirements.txt`
- For CUDA support (only on Linux), run `pip install -r cuda-requirements.txt`
- Inside the git repo, run `pip install -e .` to install the package in editable mode

## Creating new scripts
- Put a python file in `scripts` with some function `my_function`.
- The only argument to `my_function` should be an object of a dataclass `MyConfig`
  that inherits `utils.scripts.ScriptConfig`.
- The definition of `MyConfig` needs to be placed in its own file.
  (Because otherwise, serialization/deserialization doesn't work
  since the module is just `__main__`---might be worth fixing this in the future.)
- Use `utils.scripts.run(my_function, MyConfig)` in the python file to run the script.
