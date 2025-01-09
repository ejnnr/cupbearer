import torch


class HuggingfaceLM(torch.nn.Module):
    def __init__(
        self,
        tokenizer=None,
        model=None,
        tokenize_kwargs={"padding": True, "truncation": True},
        device="cuda",
    ):
        """A wrapper around a HF model that handles tokenization and device placement.

        Args:
            model: The HF model to use. May be None, in which case the model can't
                actually be used. This may be desirable if all activations are
                coming from a cache anyway.
            tokenizer: The tokenizer to use for tokenization. May be None if model
                is None.
            tokenizer_kwargs: kwargs to pass to tokenizer on forward
            device: The device to place the model on.
        """
        super().__init__()
        self.hf_model = model
        if model is not None:
            self.hf_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.tokenize_kwargs = tokenize_kwargs

        # HACK: We often use next(model.parameters()).device to figure out which
        # device a model is on. We'd like that to still work even if there's no model.
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0, device=device))

    def tokenize(self, inputs: list[str] | str, **kwargs):
        if self.tokenizer is None:
            raise ValueError("No tokenizer is set, so inputs can't be tokenized.")
        return self.tokenizer(inputs, return_tensors="pt", **kwargs).to(self.device)

    def forward(self, inputs: list[str] | str):
        if self.hf_model is None:
            raise ValueError("No model is set, so forward pass can't be run.")
        tokens = self.tokenize(inputs, **self.tokenize_kwargs)
        return self.hf_model(**tokens)

    def make_last_token_hook(self):
        """Make a hook that retrieves the activation at the last token position.

        The hook is meant to be passed to an `ActivationBasedDetector`
        as an `activation_processing_func`.
        """
        if self.tokenizer.padding_side == "left":

            def get_activation_at_last_token(
                activation: torch.Tensor, inputs: list[str], name: str
            ):
                assert activation.ndim == 3, activation.shape
                return activation[:, -1, :]

        elif self.tokenizer.padding_side == "right":

            def get_activation_at_last_token(
                activation: torch.Tensor, inputs: list[str], name: str
            ):
                # The activation should be (batch, sequence, residual dimension)
                assert activation.ndim == 3, activation.shape
                batch_size = len(inputs)

                # Tokenize the inputs to know how many tokens there are.
                # It's a bit unfortunate that we're doing this twice (once here,
                # once in the model forward pass), but not a huge deal.
                tokens = self.tokenize(inputs, **self.tokenize_kwargs)
                last_non_padding_index = tokens["attention_mask"].sum(dim=1) - 1

                return activation[range(batch_size), last_non_padding_index, :]

        else:
            raise ValueError(f"Unsupported padding side: {self.tokenizer.padding_side}")

        return get_activation_at_last_token

    def __repr__(self):
        return f"HuggingfaceLM(tokenizer={self.tokenizer}, model={self.hf_model})"
