from collections import defaultdict
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union
from contextlib import ExitStack

import torch
from torch import inference_mode
from torch.utils.data import DataLoader, Dataset

from auto_circuit.types import AblationType, SrcNode, Edge, PruneScores
from auto_circuit.utils.graph_utils import patch_mode, set_all_masks
from auto_circuit.utils.misc import remove_hooks
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.ablation_activations import src_out_hook, mean_src_out_hook
from auto_circuit.utils.graph_utils import patchable_model, set_mask_batch_size, train_mask_mode

from transformer_lens import HookedTransformer

from cupbearer import utils
from cupbearer.detectors import FeatureExtractor


StaticAblation = Literal[AblationType.ZERO, AblationType.TOKENWISE_MEAN_CLEAN]


class EAPFeatureExtractor(FeatureExtractor):
    FEATURE_NAMES = "eap_scores"
    def __init__(
            self, 
            effect_tokens: list[int],
            model: HookedTransformer,
            grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
            answer_function: Literal["avg_diff", "avg_val"],
            ablation_type: StaticAblation = AblationType.ZERO,
            abs_value: bool = False,
            integrated_grad_samples: Optional[int] = None,
            resid_src: bool = False,
            resid_dest: bool = False,
            attn: bool = True, 
            mlp: bool = False
    ):
        """
        Extracts per instance edge attribution patching scores as features, using either 
        zero or mean ablation. 
        See https://arxiv.org/abs/2310.10348 for more details on EAP, and 
        https://www.lesswrong.com/posts/caZ3yR5GnzbZe2yJ3/how-to-do-patching-fast for 
        details on the implementation. 

        Must call compute_patch_out with training dataset before calling compute_features.

        Currently only looks at metrics on final token in a sequence 
        TODO fix: currently requires branch of auto_circuit at https://github.com/oliveradk/auto-circuit/tree/oliver_dev
        
        TODO: generalize to arbitray output with loss taken on arbitary parts of the sequence 
        by taking argmax of output as effect tokens - could set effect token while 
        computing patch outs or finding the threshold
        
        Args:
            effect_tokens: The tokens to compute the score metric on
            model : The model to extract features from
            grad_function: The function 
                to apply to the logits before taking the gradient
            answer_function: The loss function of the model 
                output which the gradient is taken with respect to
            ablation_type: The type of ablation to perform. 
                Defaults to AblationType.ZERO.
            abs_value: Whether to take the absolute value of the scores. 
                Defaults to False.
            integrated_grad_samples: If not None, we compute an 
                approximation of the Integrated Gradients of the model output with respect 
                to the mask values. This is computed by averaging the mask gradients over 
                integrated_grad_samples samples of the mask values interpolated between 0 
                and 1. Cannot be used if mask_val is not None. Defaults to None.
            resid_src: Whether to use residual connections in the source 
                module. Defaults to False.
            resid_dest: Whether to use residual connections in the 
                destination module. Defaults to False.
            attn: Whether to use attention in the model. Defaults to True.
            mlp: Whether to use MLPs in the model. Defaults to False.
        """
        self.effect_tokens = torch.tensor(effect_tokens)
        self.grad_function = grad_function
        self.answer_function = answer_function
        self.ablation_type = ablation_type
        self.abs_value = abs_value
        self.integrated_grad_samples = integrated_grad_samples
        self.resid_src = resid_src
        self.resid_dest = resid_dest
        self.attn = attn
        self.mlp = mlp

        self.patch_out: torch.Tensor
        self.other_tokens: list[int]
        super().__init__(feature_names=self.FEATURE_NAMES, model=model)
    
    # should be called before model is passed
    def compute_patch_out(self, dataset: Dataset, batch_size: int):
        device = next(self.model.parameters()).device
        sample = DataLoader(
            dataset, 
            batch_size=batch_size,
            collate_fn=utils.collate_inputs
        )
        if self.ablation_type == AblationType.ZERO:
            sample = next(iter(sample))
        patch_out = src_ablations(self.model, sample, self.ablation_type, device)
        assert torch.equal(patch_out[:, 0, :, :], patch_out[:, 1, :, :])
        self.patch_out = patch_out[:, 0:1, :, :] # expand according to batch
    
    def set_model(self, model: HookedTransformer| PatchableModel):
        if hasattr(self, "model") and isinstance(self.model, PatchableModel):
            return # already set
        set_model(model)
        device = next(model.parameters()).device
        self.model: PatchableModel = patchable_model(
            model, 
            factorized=True, 
            slice_output="last_seq",
            separate_qkv=True, 
            resid_src=self.resid_src,
            resid_dest=self.resid_dest,
            attn_src=self.attn,
            attn_dest=self.attn,
            mlp_src=self.mlp,
            mlp_dest=self.mlp,
            device=device
        )
        self.other_tokens = inverse_tokens(self.effect_tokens, self.model.tokenizer.vocab_size)
    
    # not sure whether invoking the contexts on every batch incurs significant performance penalty
    def compute_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = inputs.shape[0]
        patch_out = self.patch_out.expand( # expand according to batch size (detached in method)
            self.patch_out.size(0), batch_size, self.patch_out.size(2), self.patch_out.size(3)
        )
        with ExitStack() as stack: 
            stack.enter_context(torch.enable_grad()) # if no grad is set
            # add context managers if not already in correct context
            dest_wrapper = next(iter(self.model.dest_wrappers))
            if batch_size != dest_wrapper.batch_size:
                stack.enter_context(set_mask_batch_size(self.model, batch_size))
            if not self.model.training:
                stack.enter_context(train_mask_mode(self.model))
            # compute prune scores
            prune_scores = mask_gradient_batch_prune_scores(
                model=self.model, 
                input=inputs, 
                patch_src_out=patch_out, 
                grad_function=self.grad_function, 
                answer_function=self.answer_function, 
                effect_tokens=self.effect_tokens, 
                other_tokens=self.other_tokens, 
                integrated_grad_samples=self.integrated_grad_samples
            )
            prune_scores_t = torch.concat([score.flatten(start_dim=1) for score in prune_scores.values()], dim=1)
            if self.abs_value:
                prune_scores_t = torch.abs(prune_scores_t)
            assert prune_scores_t.ndim == 2
            assert prune_scores_t.size(0) == batch_size
            return {self.FEATURE_NAMES: prune_scores_t}
    

def set_model(model: HookedTransformer, disbale_grad: bool = True):
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.eval()
    if disbale_grad:
        for param in model.parameters():
            param.requires_grad = False
    return model

def src_ablations(
    model: PatchableModel,
    sample: torch.Tensor | DataLoader,
    ablation_type: StaticAblation,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Get the activations used to ablate each [`Edge`][auto_circuit.types.Edge] in a
    model, given a particular set of model inputs and an ablation type. See
    [`AblationType`][auto_circuit.types.AblationType] for the different types of
    ablations that can be computed.

    Args:
        model: The model to get the ablations for.
        sample: The data sample to get the ablations for. This is not used for all
            `ablation_type`s. Either a single batch of inputs or a DataLoader.
        ablation_type: The type of ablation to perform.

    Returns:
        A tensor of activations used to ablated each [`Edge`][auto_circuit.types.Edge]
            model on the given input.  Shape is `[Srcs, ...]` where `Srcs` is the number
            of [`SrcNode`][auto_circuit.types.SrcNode]s in the model and `...` is the
            shape of the activations of the model. In a transformer this will be
            `[Srcs, batch, seq, d_model]`.
    """
    src_outs: Dict[SrcNode, torch.Tensor] = {}
    src_modules: Dict[torch.nn.Module, List[SrcNode]] = defaultdict(list)
    [src_modules[src.module(model)].append(src) for src in model.srcs]
    with remove_hooks() as handles, inference_mode():
        # Install hooks to collect activations at each src module
        for mod, src_nodes in src_modules.items():
            hook_fn = partial(
                mean_src_out_hook if ablation_type.mean_over_dataset else src_out_hook,
                src_nodes=src_nodes,
                src_outs=src_outs,
                ablation_type=ablation_type,
            )
            handles.add(mod.register_forward_hook(hook_fn))

        if ablation_type.mean_over_dataset:
            # Collect activations over the entire dataset and take the mean
            n = 0
            for batch in sample:
                batch = utils.inputs_from_batch(batch)
                batch = batch.to(device)
                model(batch)
                n += batch.shape[0]
            for src, src_out in src_outs.items():
                src_outs[src] = src_out / n
        else:
            # Collect activations for a single batch
            assert isinstance(sample, torch.Tensor)
            model(sample)

    # Sort the src_outs dict by node idx
    src_outs = dict(sorted(src_outs.items(), key=lambda x: x[0].src_idx))
    assert [src.src_idx for src in src_outs.keys()] == list(range(len(src_outs)))
    return torch.stack(list(src_outs.values())).detach()


def batch_answer_diff(
        vals: torch.Tensor, 
        effect_tokens: torch.Tensor, 
        other_tokens: torch.Tensor
 ) -> torch.Tensor:
    """
    Compute the difference between the values of the effect tokens and the
    other tokens in the batch.

    Args:
        vals: The values of the tokens in the batch.
        effect_tokens: The indices of the effect tokens in the batch.
        other_tokens: The indices of the other tokens in the batch.

    Returns:
        The difference between the values of the effect tokens and the other
            tokens in the batch.
    """
    bs = vals.size(0)
    effect_vals = torch.gather(vals, -1, effect_tokens.expand(bs, effect_tokens.size(0)))
    assert effect_vals.shape == (bs, effect_tokens.size(0))
    effect_vals_sum = effect_vals.sum(dim=-1)
    other_vals = torch.gather(vals, -1, other_tokens.expand(bs, other_tokens.size(0)))
    assert other_vals.shape == (bs, other_tokens.size(0))
    other_vals_sum = other_vals.sum(dim=-1)
    return effect_vals_sum - other_vals_sum


def batch_answer_val(vals: torch.Tensor, effect_tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of the values of the effect tokens in the batch

    Args:
        vals: The values of the tokens in the batch.
        effect_tokens: The indices of the effect tokens in the batch.

    Returns:
        The value of the effect tokens in the batch.
    """
    efect_vals = torch.gather(vals, -1, effect_tokens.expand(vals.size(0), effect_tokens.size(0)))
    assert efect_vals.shape == (vals.size(0), effect_tokens.size(0))
    return efect_vals.sum(dim=-1)


def mask_gradient_batch_prune_scores(
    model: PatchableModel,
    input: torch.Tensor,
    patch_src_out: torch.Tensor,
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "avg_val"],
    effect_tokens: torch.Tensor, 
    other_tokens: torch.Tensor,
    integrated_grad_samples: Optional[int] = None,
) -> PruneScores:
    """
    Same as auto_circuit.prune_algos.mask_gradient, but computes prune scores on each 
    data instance (rather than aggregating across dataset)
    Args:
        model: The model to find the circuit for.
        input: The batch input 
        grad_function: Function to apply to the logits before taking the gradient.
        answer_function: Loss function of the model output which the gradient is taken
            with respect to.
        effect_tokens: The indices of the effect tokens
        other_tokens: The indices of the other tokens

    Returns:
        Prune scores for input batch 
    """
    
    model = model
    out_slice = model.out_slice

    for sample in range((integrated_grad_samples or 0) + 1):
        # Interpolate the mask value if integrating gradients. Else set the value.
        if integrated_grad_samples is not None:
            set_all_masks(model, val=sample / integrated_grad_samples)
        else:
            set_all_masks(model, val=0)

        with patch_mode(model, patch_src_out.clone().detach()):
            logits = model(input)[out_slice]
            if grad_function == "logit":
                token_vals = logits
            elif grad_function == "prob":
                token_vals = torch.softmax(logits, dim=-1)
            elif grad_function == "logprob":
                token_vals = torch.log_softmax(logits, dim=-1)
            elif grad_function == "logit_exp":
                numerator = torch.exp(logits)
                denominator = numerator.sum(dim=-1, keepdim=True)
                token_vals = numerator / denominator.detach()
            else:
                raise ValueError(f"Unknown grad_function: {grad_function}")

            if answer_function == "avg_diff":
                loss = -batch_answer_diff(token_vals, effect_tokens, other_tokens)
            elif answer_function == "avg_val":
                loss = -batch_answer_val(token_vals, effect_tokens)
            else:
                raise ValueError(f"Unknown answer_function: {answer_function}")

            loss.backward(torch.ones_like(loss))
    prune_scores = {
        dest_wrapper.module_name: dest_wrapper.patch_mask.grad.detach().clone() 
        for dest_wrapper in model.dest_wrappers
    }
    model.zero_grad()
    return prune_scores

def inverse_tokens(tokens: torch.Tensor, vocab_size: int):
    return torch.arange(vocab_size)[~torch.isin(torch.arange(vocab_size), tokens)]