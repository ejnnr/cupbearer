import torch
import transformer_lens
from loguru import logger
from torch import nn
from transformer_lens import HookedTransformer

from .hooked_model import HookedModel


class TransformerBase(HookedModel):
    def __init__(
        self,
        model: str | HookedTransformer,
        max_length: int = 512,
        device: str | None = None,
    ):
        super().__init__()
        if isinstance(model, str):
            # Turn of LayerNorm folding etc. because we might want to finetune
            # this model.
            model = HookedTransformer.from_pretrained_no_processing(
                model, device=device
            )
        self.model: HookedTransformer = model
        self.max_length = max_length
        self._already_gave_mps_warning = False

    @property
    def default_names(self) -> list[str]:
        """Names of the activations that are returned by default."""
        return [f"blocks.{i}.hook_resid_post" for i in range(self.model.cfg.n_layers)]

    def process_input(self, x: str | list[str]) -> torch.Tensor:
        if isinstance(x, str):
            x = [x]
        elif isinstance(x, (list, tuple)):
            assert isinstance(x[0], str)
        else:
            raise ValueError(f"Expected str or list/tuple of str, got {type(x)}")

        # The type annotation for HookedTransformerConfig.device is wrong, it can
        # actually also be a device instance instead of a string (when it's set
        # automatically).
        if self.model.cfg.device in (torch.device("mps"), "mps"):
            # On MPS there's a memory leak bug if batch sizes change over training,
            # so we need to pad all inputs to the same length.
            # HookedTransformer.to_tokens doesn't support setting the padding size,
            # so we do it manually.
            # TODO: ideally we'd pad to the maximumx size in the dataset, not the
            # truncation length. This would need to happen elsewhere.

            if not self._already_gave_mps_warning:
                self._already_gave_mps_warning = True
                logger.info(
                    "To circumvent a memory leak bug on MPS, we're padding every batch "
                    "to the maximum sequence length. This may cause some slow-down or "
                    "additional memory usage compared to CPU."
                )

            # Adapted from HookedTransformer.to_tokens
            assert (
                self.model.tokenizer is not None
            ), "Cannot use to_tokens without a tokenizer"
            assert (
                self.model.cfg.tokenizer_prepends_bos is not None
            ), "Set the tokenizer for the model by calling set_tokenizer"

            if (
                self.model.cfg.default_prepend_bos
                and not self.model.cfg.tokenizer_prepends_bos
            ):
                # We want to prepend bos but the tokenizer doesn't automatically do it,
                # so we add it manually
                x = transformer_lens.utils.get_input_with_manually_prepended_bos(
                    self.model.tokenizer, x
                )

            tokens = self.model.tokenizer(
                x,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                # TODO: we could truncate to self.max_length right away but would also
                # have to do that in the main branch, so would need to overwrite
                # n_ctx. Does that have any unintended side effects?
                max_length=self.model.cfg.n_ctx,
            )["input_ids"]

            if (
                not self.model.cfg.default_prepend_bos
                and self.model.cfg.tokenizer_prepends_bos
            ):
                # We don't want to prepend bos but the tokenizer does it automatically,
                # so we remove it manually
                tokens = transformer_lens.utils.get_tokens_with_bos_removed(
                    self.model.tokenizer, tokens
                )

            tokens = tokens.to(self.model.cfg.device)
        else:
            tokens = self.model.to_tokens(x, padding_side="right", truncate=True)

        tokens = tokens[:, : self.max_length]

        if self.model.cfg.device in (torch.device("mps"), "mps"):
            assert tokens.shape[1] == self.max_length, tokens.shape
        return tokens

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s = tokens.shape

        if not self._capturing:
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter="ln_final.hook_normalized",
                return_cache_object=False,
                # Don't need the logits, we only care about the embeddings:
                return_type=None,
            )
        else:
            if self._activations != {}:
                raise ValueError("Activations already stored")

            if self._names is None:
                names = None
            else:
                names = [*self._names, "ln_final.hook_normalized"]

            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=names,
                return_cache_object=False,
                return_type=None,
            )
            # Need to modify them in place to make sure they are available
            # via the context manager.
            self._activations.update(cache)

        embeddings = cache["ln_final.hook_normalized"]
        assert embeddings.shape == (b, s, self.model.cfg.d_model), embeddings.shape
        return embeddings

    def store(self, name: str, value: torch.Tensor):
        raise NotImplementedError("store() not needed for TransformerLens based models")


class ClassifierTransformer(TransformerBase):
    def __init__(
        self,
        model: str | HookedTransformer,
        num_classes: int,
        device: str | None = None,
    ):
        super().__init__(model, device=device)
        actual_device = next(self.model.parameters()).device
        self.classifier = nn.Linear(
            self.model.cfg.d_model, num_classes, device=actual_device
        )
        # TODO: maybe we should use something less common here
        self.cls_token = torch.tensor(
            self.model.to_single_token(" omit"), device=actual_device, dtype=torch.long
        ).view(1, 1)

    def forward(self, x: str | list[str]) -> torch.Tensor:
        tokens = self.process_input(x)
        # Append CLS token:
        tokens = torch.cat([tokens, self.cls_token.expand(len(tokens), 1)], dim=1)
        embeddings = self.get_embeddings(tokens)
        logits = self.classifier(embeddings[:, -1, :])
        return logits
