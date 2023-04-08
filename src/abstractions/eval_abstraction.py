import json
from pathlib import Path
import sys
import hydra
from jax.config import config as jax_config
from loguru import logger
import sklearn.metrics

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from abstractions import abstraction, data, utils
from abstractions.computations import get_abstraction_maps
from abstractions.train_abstraction import (
    AbstractionTrainer,
    compute_losses,
    kl_loss_fn,
    single_class_loss_fn,
)


@hydra.main(version_base=None, config_path="conf", config_name="eval_abstraction")
def evaluate(cfg: DictConfig):
    """Execute model evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    train_run = Path(cfg.train_run)
    path = to_absolute_path(str(train_run / ".hydra" / "config.yaml"))
    logger.info(f"Loading abstraction config from {path}")
    train_cfg = OmegaConf.load(path)

    if cfg.debug:
        jax_config.update("jax_debug_nans", True)
        jax_config.update("jax_disable_jit", True)

    # Load the full model we want to abstract
    base_run = Path(train_cfg.base_run)
    path = to_absolute_path(str(base_run / ".hydra" / "config.yaml"))
    logger.info(f"Loading base model config from {path}")
    base_cfg = OmegaConf.load(path)

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = abstraction.Model(computation=full_computation)
    full_params = utils.load(to_absolute_path(str(base_run / "model.pytree")))["params"]

    # Magic collate_fn to get the activations of the model
    val_collate_fn = abstraction.abstraction_collate(
        full_model, full_params, return_original_batch=True
    )

    test_loaders = {}
    test_loaders["clean"] = data.get_data_loader(
        dataset=base_cfg.dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=val_collate_fn,
        transforms=data.get_transforms({}),
    )
    # For validation, we still use the training data, but with backdoors.
    # TODO: this doesn't feel very elegant.
    # Need to think about what's the principled thing to do here.
    if cfg.backdoor:
        test_loaders["backdoor"] = data.get_data_loader(
            dataset=base_cfg.dataset,
            batch_size=train_cfg.val_batch_size,
            collate_fn=val_collate_fn,
            transforms=data.get_transforms({"pixel_backdoor": {"p_backdoor": 1.0}}),
        )

    if cfg.different_corner:
        test_loaders["different_corner"] = data.get_data_loader(
            dataset=base_cfg.dataset,
            batch_size=train_cfg.val_batch_size,
            collate_fn=val_collate_fn,
            transforms=data.get_transforms(
                {"pixel_backdoor": {"p_backdoor": 1.0, "corner": "top-right"}}
            ),
        )

    if cfg.gaussian_noise:
        test_loaders["gaussian_noise"] = data.get_data_loader(
            dataset=base_cfg.dataset,
            batch_size=train_cfg.val_batch_size,
            collate_fn=val_collate_fn,
            transforms=data.get_transforms({"noise": {"std": 0.3}}),
        )

    # Dataloader returns logits, activations, and original inputs, only activations get passed to model
    _, example_activations, _ = next(iter(test_loaders["clean"]))
    # Activations are a list of batched activations, we want to effectively get
    # batch size 1
    example_input = [x[0:1] for x in example_activations]

    # TODO: the following lines are all basically copied from train_abstraction.py
    # Should deduplicate this
    if train_cfg.single_class:
        train_cfg.model.output_dim = 2

    computation = hydra.utils.call(train_cfg.model)
    # TODO: Might want to make this configurable somehow, but it's a reasonable
    # default for now
    maps = get_abstraction_maps(train_cfg.model)
    model = abstraction.Abstraction(computation=computation, abstraction_maps=maps)

    trainer = AbstractionTrainer(
        model=model,
        output_loss_fn=single_class_loss_fn if train_cfg.single_class else kl_loss_fn,
        optimizer=hydra.utils.instantiate(train_cfg.optim),
        example_input=example_input,
        log_dir=to_absolute_path(str(train_run)),
        check_val_every_n_epoch=1,
        loggers=[],
        enable_progress_bar=False,
    )
    trainer.load_model()

    metrics = trainer.eval_model(test_loaders)

    mixed_loader = data.get_data_loader(
        dataset=base_cfg.dataset,
        batch_size=cfg.max_batch_size,
        collate_fn=val_collate_fn,
        transforms=data.get_transforms({"pixel_backdoor": {"p_backdoor": 0.5}}),
    )
    logits, activations, (images, targets, infos) = next(iter(mixed_loader))
    losses = compute_losses(
        trainer.state.params,
        trainer.state,
        (logits, activations),
        output_loss_fn=trainer.output_loss_fn,
        return_batch=True,
    )
    # A higher loss should predict that the image is backdoored
    metrics["AUC_ROC"] = sklearn.metrics.roc_auc_score(
        y_true=infos["backdoored"], y_score=losses
    )

    # Print metrics to console
    trainer.on_validation_epoch_end(1, metrics, test_loaders)
    with open(train_run / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    trainer.close_loggers()


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, color="<green>", icon="📈")
    logger.add(
        sys.stderr, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stderr, filter=lambda record: record["level"].name != "METRICS")
    evaluate()
