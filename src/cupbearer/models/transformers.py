import torch
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

        tokens = self.model.to_tokens(x, padding_side="right")
        tokens = tokens[:, : self.max_length]
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
