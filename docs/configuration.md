# Configuration dataclasses
As briefly discussed in [high_level_structure.md](high_level_structure.md), `cupbearer`
makes heavy use of dataclasses for configuration. For the most part, these are just
normal dataclasses, but there are a few special things to keep in mind.

## Inherit from `BaseConfig`
All configuration dataclasses should inherit from `cupbearer.utils.utils.BaseConfig`.
That ensures that storing configs to disk and loading them again will work correctly,
as well as a debug feature described below.

Most dataclasses will not inherit from `BaseConfig` directly, but instead from a more
specialized class like `ScriptConfig` or `DatasetConfig`.

## `kw_only=True`
Sometimes a parent dataclass will have some optional fields, and then a child class will
add required fields. This would usually lead to problems because required fields can't
come after optional ones. To deal with that, many dataclasses in `cupbearer` use
`@dataclass(kw_only=True)`, which makes all fields keyword-only arguments to `__init__`.

## `_set_debug()`
It can be convenient to run a script with the fastest possible settings for debugging
error messages or for automated testing (e.g. just train for a single batch with
a single sample, use a small model, ...). In `cupbearer`, every configuration dataclass
should "know" how to set itself to such a debug mode: it should have a `_set_debug()`
method that sets all its fields to the debug values that lead to fast runs. Of course
if a config has no such values, it doesn't need to implement `_set_debug()`.

Importantly, `_set_debug()` should also call `super()._set_debug()`. This ensures that
fields from the parent class are set to their debug values. It also recursively calls
`_set_debug()` on all fields that are themselves configuration dataclasses, so there's
no need to do that manually.

## Special CLI fields
You can use `simple_parsing.helpers.field` instead of the builtin `dataclasses.field`
to get some additional functionality, most notably specifying how options can be changed
from the CLI. This will mostly be unnecessary, but can be nice for boolean flags.

For example, the debug option described above is implemented using
```python
debug: bool = field(action="store_true")
```
in `ScriptConfig`, which means you can call scripts using simply `--debug` instead of
`--debug True`.
