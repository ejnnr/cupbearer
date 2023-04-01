# based on the flax example code

import sys
from typing import Callable, Optional
import hydra
import jax
import jax.numpy as jnp
from jax.config import config as jax_config
from loguru import logger

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from abstractions import abstraction, data, train_mnist, utils, trainer
from abstractions.logger import WandbLogger, DummyLogger


def kl_loss_fn(predicted_logits, logits):
    # KL divergence between actual and predicted output:
    # TODO: Should probably just use something like distrax.
    probs = jax.nn.softmax(logits, axis=-1)
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    predicted_logprobs = jax.nn.log_softmax(predicted_logits, axis=-1)
    output_losses = (probs * (logprobs - predicted_logprobs)).sum(axis=-1)
    return output_losses


def single_class_loss_fn(predicted_logits, logits, target=0):
    # This loss function only cares about correctly predicting whether the output
    # is a specific digit (like zero) or not. It combines the probabilities for all non-zero classes into
    # a single "non-zero" class. Assumes that output_dim of the abstraction is 2.
    assert predicted_logits.ndim == logits.ndim == 2
    assert logits.shape[-1] == 10, logits.shape
    assert predicted_logits.shape[-1] == 2, predicted_logits.shape
    assert predicted_logits.shape[0] == logits.shape[0]

    all_probs = jax.nn.softmax(logits, axis=-1)
    target_probs = all_probs[:, target]
    non_target_probs = all_probs[:, :target].sum(axis=-1) + all_probs[
        :, target + 1 :
    ].sum(-1)
    probs = jnp.stack([non_target_probs, target_probs], axis=-1)
    logprobs = jnp.log(probs)
    predicted_logprobs = jax.nn.log_softmax(predicted_logits, axis=-1)
    output_losses = (probs * (logprobs - predicted_logprobs)).sum(axis=-1)
    return output_losses


class AbstractionTrainer(trainer.TrainerModule):
    def __init__(
        self,
        output_loss_fn: Callable = kl_loss_fn,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        # The output loss function describes how to compute the loss between the
        # actual output and the predicted one. It should take a batch of predicted
        # outputs and a batch of actual outputs and return a *batch* of losses.
        # The main point is to allow ignoring certain differences in the output
        # that we aren't aiming to predict
        self.output_loss_fn = output_loss_fn

    def create_functions(self):
        def losses(params, state, batch, mask=None):
            logits, activations = batch
            abstractions, predicted_abstractions, predicted_logits = state.apply_fn(
                {"params": params}, activations
            )
            assert isinstance(abstractions, list)
            assert isinstance(predicted_abstractions, list)
            assert len(abstractions) == len(predicted_abstractions) + 1
            b, *_ = abstractions[0].shape
            assert logits.shape == (b, 10)
            # Output dimension of abstraction may be different from full computation,
            # depending on the output_loss_fn. So only check batch dimension.
            assert predicted_logits.shape[0] == b

            if mask is None:
                mask = jnp.ones((b,))
            assert mask.shape == (b,)
            num_samples = mask.sum()

            output_losses = self.output_loss_fn(predicted_logits, logits)
            output_losses *= mask
            # Can't take mean here because we don't want to count masked samples
            output_loss = output_losses.sum() / num_samples
            # Consistency loss:
            consistency_loss = jnp.array(0)
            # Skip the first abstraction, since there's no prediction for that
            for abstraction, predicted_abstraction in zip(
                abstractions[1:], predicted_abstractions
            ):
                # Take mean over hidden dimension(s):
                consistency_losses = ((abstraction - predicted_abstraction) ** 2).mean(
                    axis=tuple(range(1, abstraction.ndim))
                )
                consistency_losses *= mask
                # Now we also take the mean over the batch (outside the sqrt and after
                # masking). As before, we don't want to count masked samples.
                consistency_loss += jnp.sqrt(consistency_losses).sum() / num_samples

            consistency_loss /= len(predicted_abstractions)

            loss = output_loss + consistency_loss

            return loss, (output_loss, consistency_loss)

        def train_step(state, batch):
            loss_fn = lambda params: losses(params, state, batch)
            (loss, (output_loss, consistency_loss)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)
            param_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), state.params)
            max_param_norm = jnp.max(jnp.array(jax.tree_util.tree_leaves(param_norms)))

            grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), grads)
            max_grad_norm = jnp.max(jnp.array(jax.tree_util.tree_leaves(grad_norms)))

            state = state.apply_gradients(grads=grads)
            metrics = {
                "loss": loss,
                "output_loss": output_loss,
                "consistency_loss": consistency_loss,
                "max_param_norm": max_param_norm,
                "max_grad_norm": max_grad_norm,
            }
            return state, metrics

        def eval_step(state, batch):
            # Note that this class expects the eval dataloader to return not just
            # logits/activations, but also the original data. That's necessary
            # to compute metrics on specific clases (zero in this case).
            logits, activations, (images, labels, infos) = batch
            loss, (output_loss, consistency_loss) = losses(
                state.params, state, (logits, activations)
            )
            zeros = infos["original_target"] == 0
            zeros_loss, (zeros_output_loss, zeros_consistency_loss) = losses(
                state.params, state, (logits, activations), mask=zeros
            )
            metrics = {
                "loss": loss,
                "output_loss": output_loss,
                "consistency_loss": consistency_loss,
                "zeros_loss": zeros_loss,
                "zeros_output_loss": zeros_output_loss,
                "zeros_consistency_loss": zeros_consistency_loss,
            }
            return metrics

        return train_step, eval_step

    def on_training_epoch_end(self, epoch_idx, metrics):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def on_validation_epoch_end(self, epoch_idx: int, metrics, val_loader):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())


@hydra.main(version_base=None, config_path="conf", config_name="abstraction")
def train_and_evaluate(cfg: DictConfig):
    """Execute model training and evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    if cfg.debug:
        jax_config.update("jax_debug_nans", True)
        jax_config.update("jax_disable_jit", True)
        cfg.wandb = False

    if cfg.wandb:
        metrics_logger = WandbLogger(
            project_name="abstractions",
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )
    else:
        metrics_logger = DummyLogger()

    # Load the full model we want to abstract
    model = hydra.utils.instantiate(cfg.model)
    params = utils.load(to_absolute_path(cfg.model_path))["params"]

    # Magic collate_fn to get the activations of the model
    train_collate_fn = abstraction.abstraction_collate(model, params)
    val_collate_fn = abstraction.abstraction_collate(
        model, params, return_original_batch=True
    )

    train_loader = data.get_data_loaders(
        cfg.batch_size,
        collate_fn=train_collate_fn,
        transforms=data.get_transforms({"pixel_backdoor": {"p_backdoor": 0.0}}),
    )
    # For validation, we still use the training data, but with backdoors.
    # TODO: this doesn't feel very elegant.
    # Need to think about what's the principled thing to do here.
    backdoor_loader = data.get_data_loaders(
        cfg.batch_size,
        collate_fn=val_collate_fn,
        transforms=data.get_transforms({"pixel_backdoor": {"p_backdoor": 1.0}}),
    )
    val_loaders = {
        "backdoor": backdoor_loader,
    }

    # Dataloader returns logits and activations, only activations get passed to model
    _, example_activations = next(iter(train_loader))
    # Activations are a list of batched activations, we want to effectively get
    # batch size 1
    example_input = [x[0:1] for x in example_activations]

    if cfg.single_class:
        cfg.abstraction.output_dim = 2

    trainer = AbstractionTrainer(
        model=hydra.utils.instantiate(cfg.abstraction),
        output_loss_fn=single_class_loss_fn if cfg.single_class else kl_loss_fn,
        optimizer=hydra.utils.instantiate(cfg.optim),
        example_input=example_input,
        # Hydra sets the cwd to the right log dir automatically
        log_dir=".",
        check_val_every_n_epoch=1,
        loggers=[metrics_logger],
        enable_progress_bar=False,
    )

    trainer.train_model(
        train_loader=train_loader,
        val_loaders=val_loaders,
        test_loaders=None,
        num_epochs=cfg.num_epochs,
    )
    trainer.save_model()
    trainer.close_loggers()


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, color="<green>", icon="📈")
    logger.add(
        sys.stderr, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stderr, filter=lambda record: record["level"].name != "METRICS")
    train_and_evaluate()
