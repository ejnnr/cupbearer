# Adapted from https://github.com/wanlunsec/Beatrix/blob/master/defenses/Beatrix/Beatrix.py#L307
# Reference: "The Beatrix Resurrections: Robust Backdoor Detection via Gram Matrices" [https://arxiv.org/abs/2209.11715v3].
import torch
from einops import einsum, rearrange
from loguru import logger

from cupbearer.detectors.statistical.statistical import StatisticalDetector


class BeatrixDetector(StatisticalDetector):
    """Beatrix detector that uses Gram matrices and mean absolute deviation statistics
    for anomaly detection.
    Reference: "The Beatrix Resurrections: Robust Backdoor Detection via Gram Matrices"
    [https://arxiv.org/abs/2209.11715v3].
    """

    def __init__(
        self,
        power_list=None,
        mad_scale=10.0,
        sequence_dim_as_batch=True,
        moving_average=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.power_list = power_list or list(range(1, 5))  # List of powers to compute
        self.mad_scale = mad_scale  # Scale factor for the median absolute deviation
        self._stats = {}  # Stores running statistics for Gram features
        # _stats[case][layer_name][power]["n_samples"] = int
        # _stats[case][layer_name][power]["running_medians"] = List[Tensor(
        #     n_gram_features,)]
        # _stats[case][layer_name][power]["running_mads"] = List[Tensor(
        #     n_gram_features,)]

        # Whether to treat all dims (exept the last) as batch:
        self.sequence_dim_as_batch = sequence_dim_as_batch
        # Whether to usage moving average for running statistics
        #  or to compute the median of on-batch medians
        self.moving_average = moving_average
        self.eps = torch.finfo(torch.float16).tiny  # Small value to avoid zero-division

    def compute_gram_features(
        self, activation: torch.Tensor, power: int
    ) -> torch.Tensor:
        """Compute p-th order Gram features for given activation features.

        Args:
            features: Activation features of shape (batch, dim)
            power: Power to raise features to before computing Gram matrix

        Returns:
            Vectorized upper triangular elements of Gram matrix
        """
        # Reshape to (batch, features) treating all other dims as batch
        if self.sequence_dim_as_batch:
            features = rearrange(activation, "batch ... dim -> (batch ...) dim")
        else:
            features = rearrange(activation, "batch ... dim -> batch (...) dim")

        # Compute p-th power
        powered_features = features**power

        # Compute Gram matrix
        gram = einsum(
            powered_features,
            powered_features,
            "batch ... dim_a, batch ... dim_b -> batch dim_a dim_b",
        )
        # "..." contains nothing if sequence_dim_as_batch, else it's the sequence dim.

        # Apply p-th root
        gram = gram.sign() * torch.abs(gram) ** (1.0 / power)

        # Get upper triangular elements (excluding diagonal)
        triu_indices = torch.triu_indices(gram.size(-2), gram.size(-1))
        gram_vector = gram[..., triu_indices[0], triu_indices[1]]

        gram_vector = torch.nan_to_num(gram_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return gram_vector  # (batch, n_gram_features)

    def init_variables(
        self,
        sample_batch,
        case: str,
    ):
        _, example_activations = sample_batch

        # v is an entire batch, v[0] are activations for a single input
        activation_sizes = {k: v[0].size() for k, v in example_activations.items()}

        """Initialize statistical variables for training."""
        if any(len(size) != 1 for size in activation_sizes.values()):
            logger.debug(
                "Received multi-dimensional activations, will only take products for"
                "the gram matrix along last dimension and treat others independently. "
                "If this is unintentional, pass "
                "`activation_preprocessing_func=utils.flatten_last`."
            )
        logger.debug(
            "Activation sizes: \n"
            + "\n".join(f"{k}: {size}" for k, size in activation_sizes.items())
        )

        self._stats[case] = {
            layer_name: {
                p: {
                    "n_samples": 0,
                    "running_medians": [],  # Will be initialized on first batch
                    "running_mads": [],
                }
                for p in self.power_list
            }
            for layer_name in activation_sizes.keys()
        }

    def update_stats(self, current_stats: dict, gram_features: torch.Tensor):
        """Update running median and MAD statistics for Gram features."""
        n = current_stats["n_samples"]
        total_n = n + len(gram_features)

        # Update medians
        new_medians = gram_features.median(dim=0).values

        # Update median absolute deviations
        deviations = torch.abs(gram_features - new_medians)
        new_mads = deviations.median(dim=0).values

        return {
            "n_samples": total_n,  # int
            "running_medians": current_stats["running_medians"] + [new_medians.cpu()],
            "running_mads": current_stats["running_mads"] + [new_mads.cpu()],
            # ^ Both List[shape (n_gram_features,)]
        }

    def update_stats_moving_average(
        self, current_stats: dict, gram_features: torch.Tensor
    ):
        """Update running median and MAD statistics for Gram features, using an
        exponential moving average."""
        if len(current_stats["running_medians"]) == 0:
            # Initialize on first batch
            current_stats["running_medians"] = [gram_features.median(dim=0).values]
            current_stats["running_mads"] = [
                (
                    torch.abs(gram_features - current_stats["running_medians"][0])
                    .median(dim=0)
                    .values
                )
            ]
            current_stats["n_samples"] = len(gram_features)
            return current_stats

        n = current_stats["n_samples"]
        total_n = n + len(gram_features)

        # Update running median using exponential moving average
        # For small batch sizes, this is more like a mean than a median.
        alpha = len(gram_features) / total_n
        new_medians = (1 - alpha) * current_stats["running_medians"][
            0
        ] + alpha * gram_features.median(dim=0).values

        # Update median absolute deviations
        deviations = torch.abs(gram_features - new_medians)
        new_mads = (1 - alpha) * current_stats["running_mads"][
            0
        ] + alpha * deviations.median(dim=0).values

        return {
            "n_samples": total_n,  # int
            "running_medians": [new_medians],  # shape (n_gram_features,)
            "running_mads": [new_mads],  # shape (n_gram_features,)
        }

    def batch_update(self, activations: dict[str, torch.Tensor], case: str):
        """Update statistics with new batch of activations."""
        for layer_name, activation in activations.items():
            for power in self.power_list:
                # Compute Gram features for this batch
                gram_features = self.compute_gram_features(activation, power)

                # Update running statistics
                if self.moving_average:
                    self._stats[case][layer_name][
                        power
                    ] = self.update_stats_moving_average(
                        self._stats[case][layer_name][power], gram_features
                    )
                else:
                    self._stats[case][layer_name][power] = self.update_stats(
                        self._stats[case][layer_name][power], gram_features
                    )

    def _compute_layerwise_scores(self, inputs, features: dict[str, torch.Tensor]):
        """Compute anomaly scores for each layer."""
        scores = {}
        next(iter(features.values())).shape[0]

        for layer_name, activation in features.items():
            layer_scores = []

            for power in self.power_list:
                stats = self.stats["trusted"][layer_name][power]

                # Compute Gram features for test sample
                # shape (batch, n_gram_features)
                gram_features = self.compute_gram_features(activation, power)

                medians = stats["medians"].to(gram_features.device)
                mads = stats["mads"].to(gram_features.device)

                print(gram_features.shape)

                # Compute min/max bounds
                # shape (n_gram_features,)
                min_bounds = medians - self.mad_scale * mads
                max_bounds = medians + self.mad_scale * mads

                # Compute deviations
                # shape (batch, n_gram_features)
                lower_devs = torch.relu(min_bounds - gram_features) / (
                    torch.abs(min_bounds) + self.eps
                )

                upper_devs = torch.relu(gram_features - max_bounds) / (
                    torch.abs(max_bounds) + self.eps
                )

                deviations = lower_devs + upper_devs
                # Average across Gram matrix elements
                # shape (batch,)
                mean_deviation = deviations.mean(dim=-1)
                layer_scores.append(mean_deviation)

            # Average scores across different powers
            # shape (batch,)
            scores[layer_name] = torch.stack(layer_scores, dim=-1).mean(dim=-1)

        return scores

    def _finalize_training(self, **kwargs):
        # Approximate the true medians as the medians across batches
        self.stats = {}
        for case in self._stats:
            self.stats[case] = {}
            for layer_name in self._stats[case]:
                self.stats[case][layer_name] = {}
                for power in self._stats[case][layer_name]:
                    self.stats[case][layer_name][power] = {
                        "n_samples": self._stats[case][layer_name][power]["n_samples"],
                        "medians": torch.stack(
                            self._stats[case][layer_name][power]["running_medians"]
                        )
                        .median(dim=0)
                        .values,
                        "mads": torch.stack(
                            self._stats[case][layer_name][power]["running_mads"]
                        )
                        .median(dim=0)
                        .values,
                    }

    def _get_trained_variables(self):
        """Return variables needed for inference."""
        return {
            "stats": self.stats,
        }

    def _set_trained_variables(self, variables):
        """Set trained variables for inference."""
        self.stats = variables["stats"]

    def __repr__(self):
        return "BeatrixDetector()"
