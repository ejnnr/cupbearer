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
        max_length: int | None = None,
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
        self.max_length = max_length or self.model.cfg.n_ctx
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
            tokens = self.model.to_tokens(x, truncate=True)

        tokens = tokens[:, : self.max_length]

        if self.model.cfg.device in (torch.device("mps"), "mps"):
            assert tokens.shape[1] == self.max_length, tokens.shape
        return tokens

    def get_embeddings(
        self, tokens: torch.Tensor, layer: int | None = None
    ) -> torch.Tensor:
        b, s = tokens.shape
        # We handle the embedding layer manually because we want the model to get
        # an attention mask, which means we have to either use start_at_layer, or
        # pass in strings to forward(). But we want to be able to pass in tokens
        # so that we can easily add a classification token at the end.
        #
        # This call shouldn't do any padding, but we need to specify the padding side
        # to force it to compute attention masks.
        (
            residual,
            _,
            shortformer_pos_embed,
            attention_mask,
        ) = self.model.input_to_embed(tokens)

        embedding_name = (
            f"blocks.{layer}.hook_resid_post"
            if layer is not None
            else "ln_final.hook_normalized"
        )

        # By default (if not capturing intermediate activations), only capture
        # the layer we use for logit computations:
        names = [embedding_name]

        if self._capturing:
            if self._activations != {}:
                raise ValueError("Activations already stored")

            if self._names is None:
                # capture everything
                names = None
            else:
                # capture only the specified names, plus the one we need later
                # to compute logits
                names += self._names

        _, cache = self.model.run_with_cache(
            input=residual,
            attention_mask=attention_mask,
            # skip the embedding layer, we handled that manually
            start_at_layer=0,
            names_filter=names,
            return_cache_object=False,
            # Don't need the logits, we only care about the embeddings:
            return_type=None,
        )

        if self._capturing:
            # Need to modify them in place to make sure they are available
            # via the context manager.
            self._activations.update(cache)

        embeddings = cache[embedding_name]
        assert embeddings.shape == (b, s, self.model.cfg.d_model), embeddings.shape
        return embeddings

    def store(self, name: str, value: torch.Tensor):
        raise NotImplementedError("store() not needed for TransformerLens based models")


class ClassifierTransformer(TransformerBase):
    def __init__(
        self,
        model: str | HookedTransformer,
        num_classes: int,
        freeze_model: bool = False,
        device: str | None = None,
    ):
        super().__init__(model, device=device)
        # We need to set this on the tokenizer, just passing it to `input_to_embeds`
        # doesn't work, since that function only checks the padding_side on the
        # tokenizer to determine whether to compute an attention mask.
        # (It doesn't use LocallyOverridenDefaults itself, presumably since it's mainly
        # meant to be called from forward(), which does that already.)
        self.model.tokenizer.padding_side = "left"

        self.freeze_model = freeze_model
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        actual_device = next(self.model.parameters()).device
        self.classifier = nn.Linear(
            self.model.cfg.d_model, num_classes, device=actual_device
        )
        # TODO: maybe we should use something less common here, this is just takend
        # from Redwood's measurement tampering benchmark.
        # TODO: figure out whether we want to use this or not---right now it's
        # unnecessary because we're just summing over all tokens anyway.
        # self.cls_token = torch.tensor(
        #     self.model.to_single_token(" omit"), device=actual_device,
        #     dtype=torch.long
        # ).view(1, 1)

    def forward(self, x: str | list[str]) -> torch.Tensor:
        tokens = self.process_input(x)
        # TODO: remove if we don't use this
        # if tokens.shape[1] == self.max_length:
        #     # Make space for the CLS token we'll add
        #     tokens = tokens[:, : self.max_length - 1]
        # Append CLS token:
        # tokens = torch.cat([tokens, self.cls_token.expand(len(tokens), 1)], dim=1)

        if self.freeze_model:
            # If we're freezing the model, we don't use final layer embeddings since
            # these have to correspond directly to next token predictions.
            # TODO: does this really matter, and is the penultimate layer best?
            # The - 2 is because we're zero-indexing layers.
            embeddings = self.get_embeddings(tokens, layer=self.model.cfg.n_layers - 2)
        else:
            embeddings = self.get_embeddings(tokens)

        # Take the mean over the sequence dimension, but ignore padding tokens.
        # We're mainly doing this for the case where the model is frozen. If it's not,
        # then this isn't really necessary but also shouldn't hurt.
        mask = tokens != self.model.tokenizer.pad_token_id
        mask = mask.unsqueeze(-1)
        assert mask.shape == tokens.shape + (1,)
        assert embeddings.shape == tokens.shape + (self.model.cfg.d_model,)
        embeddings = embeddings * mask
        embeddings = embeddings.sum(dim=1) / mask.sum(dim=1)

        logits = self.classifier(embeddings)
        return logits
