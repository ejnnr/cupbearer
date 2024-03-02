from typing import Callable


def script(
    script_fn: Callable,
) -> Callable:
    # @functools.wraps(script_fn)
    # def run_script(cfg: ConfigType):
    #     save_cfg(cfg, save_config=cfg.save_config)
    #     return script_fn(cfg)

    # return run_script
    return script_fn


def save_cfg(cfg, save_config: bool = True):
    # if cfg.path:
    #     cfg.path.mkdir(parents=True, exist_ok=True)
    #     if save_config:
    #         # TODO: replace this with cfg.save if/when that exposes save_dc_types.
    #         # Note that we need save_dc_types here even though `BaseConfig` already
    #         # enables that, since `save` calls `to_dict` directly, not `obj.to_dict`.
    #         simple_parsing.helpers.serialization.serializable.save(
    #             cfg,
    #             cfg.path / "config.yaml",
    #             save_dc_types=True,
    #             sort_keys=False,
    #         )
    pass
