import torch


class HuggingfaceLM(torch.nn.Module):
    def __init__(self, hf_model, tokenizer, device="cuda"):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.device = device

    def tokenize(self, inputs: list[str] | str):
        return self.tokenizer(inputs, padding=True, return_tensors="pt").to(self.device)

    def forward(self, inputs: list[str] | str):
        tokens = self.tokenize(inputs)
        return self.hf_model(**tokens)

    def make_last_token_hook(self):
        """Make a hook that retrieves the activation at the last token position.

        The hook is meant to be passed to an `ActivationBasedDetector`
        as an `activation_processing_func`.
        """

        def get_activation_at_last_token(
            activation: torch.Tensor, inputs: list[str], name: str
        ):
            # The activation should be (batch, sequence, residual dimension)
            assert activation.ndim == 3, activation.shape
            assert activation.shape[-1] == 4096, activation.shape
            batch_size = len(inputs)

            # Tokenize the inputs to know how many tokens there are.
            # It's a bit unfortunate that we're doing this twice (once here,
            # once in the model forward pass), but not a huge deal.
            tokens = self.tokenize(inputs)
            last_non_padding_index = tokens["attention_mask"].sum(dim=1) - 1

            return activation[range(batch_size), last_non_padding_index, :]

        return get_activation_at_last_token
