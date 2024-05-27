# Developer guide
## Setup
1. Clone this git repository
2. Create a virtual environment with Python 3.10 and activate it
3. Run `pip install -e .[dev]` inside the git repo to install the package in editable mode
   with development dependencies
4. Run `pre-commit install` to install the pre-commit git hooks (this will lint and
   often auto-fix your code before every commit)
5. Add the following to your `.git/config` to automatically strip output and metadata
   from notebooks before committing them. This will *not* modify the notebooks on your
   disk, just what's committed to git.
   ```
   [filter "strip-notebook-output"]
      clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
   ```

## Tests
Run `pytest` to run all tests. New tests should be added in the `tests` directory.

## Understanding `cupbearer`
The [high-level structure](high_level_structure.md) document gives a brief overview of the
different subpackages in `cupbearer` and is a good place to start. From there you can
dive deeper by reading the other files in this folder, depending on what kind of additions
you want to make to `cupbearer`.
