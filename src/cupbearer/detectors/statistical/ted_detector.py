from typing import Dict, Literal, Union

import torch
from pyod.models.pca import PCA
from tqdm.auto import tqdm

from cupbearer.detectors.statistical.statistical import StatisticalDetector


class TEDDetector(StatisticalDetector):
    """Topological Evolution Dynamics detector that tracks activation trajectories
    across layers.
    Adapted from Robust Backdoor Detection for Deep Learning via Topological Evolution
    Dynamics https://arxiv.org/abs/2312.02673

    This detector examines how the ranking of nearest neighbors changes across layers
    to detect anomalies. For each input:
    1. Find its k-nearest neighbors in a reference layer
    2. Track how each neighbor's ranking changes through other layers
    3. Use PCA-based outlier detection on these ranking trajectories

    Unlike the original paper which focused on classification tasks and tracked nearest
    neighbors within predicted classes, this version is adapted for generative models
    and tracks nearest neighbors among all clean samples.

    Args:
        n_neighbors: Number of nearest neighbors to track for each sample
        contamination: Proportion of samples to consider as outliers during training
        normalize_ranks: Whether to normalize ranks to [0,1] range
        score_aggregation: How to combine neighbors anomaly scores ("mean" or "max")
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        contamination: float = 0.1,
        normalize_ranks: bool = False,
        score_aggregation: str = "mean",
        store_acts_on_cpu: bool = True,
        sequence_dim_as_batch: bool = False,
        max_seq_len: Union[int, None] = None,
        truncate_seq_at: Literal["start", "end"] = "start",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.normalize_ranks = normalize_ranks
        self.sequence_dim_as_batch = sequence_dim_as_batch
        self.score_aggregation = score_aggregation
        self.store_acts_on_cpu = store_acts_on_cpu
        self.max_seq_len = max_seq_len
        self.truncate_seq_at = truncate_seq_at

        # Storage for training state
        self.clean_activations = {}  # Layer name -> List[Tensor]
        self.pca_detectors = {}  # Layer name -> PCA detector

    def init_variables(self, sample_batch, case: str):
        """Initialize storage for clean activations."""
        _, example_activations = sample_batch

        # Initialize empty lists to store activations from each layer
        self.clean_activations = {
            layer_name: [] for layer_name in example_activations.keys()
        }

        self.max_seq_len_seen = {
            layer_name: 0 for layer_name in example_activations.keys()
        }

        # Counter for number of samples seen
        self._ns = {case: 0 for case in ["trusted", "untrusted"]}

    def batch_update(self, activations: Dict[str, torch.Tensor], case: str):
        """Store clean activations from this batch.

        For transformer models, activations will typically have shape:
        (batch_size, sequence_length, hidden_dim)
        """
        if case != "trusted":
            return

        # Store activations from each layer
        # Note: The positions in these lists implicitly track which
        # activations came from the same sample
        for layer_name, activation in activations.items():
            if activation.ndim == 3:
                self.max_seq_len_seen[layer_name] = max(
                    self.max_seq_len_seen[layer_name], activation.size(1)
                )
            if self.store_acts_on_cpu:
                activation = activation.cpu()
            self.clean_activations[layer_name].append(activation)

        # Update sample counter
        self._ns[case] += next(iter(activations.values())).size(0)

    def _find_k_nearest(
        self, query: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Find indices of k nearest neighbors by cosine similarity.

        Args:
            query: Shape (n_queries, hidden_dim)
            reference: Shape (n_reference, hidden_dim)

        Returns:
            Indices of shape (n_queries, k)
        """
        # Normalize for cosine similarity
        query_norm = torch.nn.functional.normalize(query, p=2, dim=1)
        ref_norm = torch.nn.functional.normalize(reference, p=2, dim=1)

        # Compute similarities and find top k
        similarities = torch.mm(query_norm, ref_norm.t())
        _, indices = similarities.topk(k=self.n_neighbors, dim=1)
        return indices

    def _get_neighbor_rankings(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Find ranking of specific neighbors among all reference samples.

        Args:
            query: Shape (n_queries, hidden_dim)
            reference: Shape (n_reference, hidden_dim)
            neighbor_indices: Shape (n_queries, k)

        Returns:
            Rankings of shape (n_queries, k)
        """

        # Normalize for cosine similarity
        query_norm = torch.nn.functional.normalize(query, p=2, dim=1)
        ref_norm = torch.nn.functional.normalize(reference, p=2, dim=1)

        # Compute similarities between queries and reference samples
        similarities = torch.mm(query_norm, ref_norm.t())

        # For each query, find rank of its neighbors among all samples
        neighbor_rankings = torch.zeros_like(neighbor_indices, dtype=torch.float32)

        for i in range(len(query)):
            # Sort similarities for this query to get rankings
            _, sorted_indices = torch.sort(similarities[i], descending=True)
            # Convert to ranks (position in sorted list)
            ranks = torch.zeros_like(
                sorted_indices, dtype=torch.float32, device=query.device
            )
            ranks[sorted_indices] = torch.arange(
                len(sorted_indices), dtype=torch.float32, device=query.device
            )

            # Look up ranks of this query's neighbors
            query_neighbors = neighbor_indices[i]
            neighbor_rankings[i] = ranks[query_neighbors]

        if self.normalize_ranks:
            # Normalize to [0,1] range
            neighbor_rankings = neighbor_rankings / (reference.size(0) - 1)

        return neighbor_rankings

    def _prepare_activation(
        self, activation: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        """Prepare activations for storage.

        For transformer models, activations will typically have shape:
        (batch_size, sequence_length, hidden_dim)
        """
        act = activation.clone().detach()

        # If we have a sequence dimension
        if act.ndim > 2:
            # Reverse sequence if needed
            if self.truncate_seq_at == "start":
                act = act.flip(1)
            # Truncate sequence if needed
            if self.max_seq_len is not None:
                max_seq_len = min(self.max_seq_len, self.max_seq_len_seen[layer_name])
            else:
                max_seq_len = self.max_seq_len_seen[layer_name]
            if act.size(1) > max_seq_len:
                act = act[:, :max_seq_len, :]
            else:  # Pad with zeros otherwise
                pad_size = max_seq_len - act.size(1)
                act = torch.cat(
                    [
                        act,
                        torch.zeros(
                            act.size(0), pad_size, act.size(2), device=act.device
                        ),
                    ],
                    dim=1,
                )

            if self.sequence_dim_as_batch:
                # Flatten sequence into batch dimension
                act = act.reshape(-1, activation.size(-1))
            else:
                # Flatten sequence into hidden dimension
                act = act.reshape(activation.size(0), -1)
        return act.to(float)

    def _finalize_training(self, **kwargs):
        """Create PCA outlier detectors for each reference layer."""
        # Stack stored activations for each layer
        for layer_name in self.clean_activations:
            for i in range(len(self.clean_activations[layer_name])):
                self.clean_activations[layer_name][i] = self._prepare_activation(
                    self.clean_activations[layer_name][i], layer_name
                )

            self.clean_activations[layer_name] = torch.cat(
                self.clean_activations[layer_name], dim=0
            )

        layer_names = list(self.clean_activations.keys())

        # Get reference neighbors in each reference layer
        ref_neighbors = {}
        for ref_layer in tqdm(layer_names, desc="Finding reference neighbors"):
            ref_activations = self.clean_activations[ref_layer]
            similarities = torch.mm(
                torch.nn.functional.normalize(ref_activations, p=2, dim=1),
                torch.nn.functional.normalize(ref_activations, p=2, dim=1).t(),
            )
            _, neighbor_indices = similarities.topk(k=self.n_neighbors + 1, dim=1)
            ref_neighbors[ref_layer] = neighbor_indices[:, 1:]  # Skip self

        # For each reference layer
        for ref_layer in tqdm(layer_names, desc="Creating PCA detectors"):
            # Get selected neighbors from reference layer
            neighbors = ref_neighbors[ref_layer]

            # Track ranking trajectories for these neighbors through all layers
            ranking_vectors = []
            # For each neighbor
            for k in range(self.n_neighbors):
                k_neighbors = neighbors[:, k]
                layer_rankings = []

                # Track rankings through layers
                for layer in layer_names:
                    if layer != ref_layer:
                        layer_activations = self.clean_activations[layer]
                        rankings = self._get_neighbor_rankings(
                            query=layer_activations,
                            reference=layer_activations,
                            neighbor_indices=k_neighbors.unsqueeze(1),
                        ).squeeze(1)
                        layer_rankings.append(rankings)

                ranking_vectors.append(torch.stack(layer_rankings, dim=1))

            # Stack vectors from different neighbors # (n_samples, k, n_layers)
            ranking_vectors = torch.stack(ranking_vectors, dim=1)

            # Fit PCA detector
            ranking_vectors = ranking_vectors.reshape(-1, len(layer_names) - 1)

            pca = PCA(contamination=self.contamination)
            pca.fit(ranking_vectors.cpu().numpy())
            self.pca_detectors[ref_layer] = pca

    def _compute_layerwise_scores(
        self, inputs: tuple, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute anomaly scores based on topological evolution.

        For each input and each reference layer:
        1. Find k nearest clean neighbors in reference layer
        2. Track rankings of those neighbors through all layers
        3. Use PCA detector to score the ranking trajectories
        4. Aggregate scores across neighbors and sequence positions
        """
        example_features = next(iter(features.values()))
        device = example_features.device
        batch_size = example_features.size(0)
        layer_names = list(features.keys())

        prepared_features = {}
        for layer in layer_names:
            prepared_features[layer] = self._prepare_activation(features[layer], layer)

        # Compute scores using each layer as reference
        all_scores = {}
        for ref_layer in layer_names:
            # Prepare query activations (combine batch and sequence dims)
            query = prepared_features[ref_layer]
            clean_ref = self.clean_activations[ref_layer].to(device)

            # Find k nearest neighbors in reference layer
            neighbor_indices = self._find_k_nearest(query, clean_ref)

            # Track rankings through layers
            ranking_vectors = []
            for k in range(self.n_neighbors):
                k_neighbors = neighbor_indices[:, k]

                layer_rankings = []
                for layer in layer_names:
                    if layer != ref_layer:
                        # Get rankings in this layer
                        layer_query = prepared_features[layer]
                        layer_clean = self.clean_activations[layer].to(device)
                        rankings = self._get_neighbor_rankings(
                            query=layer_query,
                            reference=layer_clean,
                            neighbor_indices=k_neighbors.unsqueeze(1),
                        ).squeeze(1)
                        layer_rankings.append(rankings)

                ranking_vectors.append(torch.stack(layer_rankings, dim=1))

            # Stack across neighbors
            ranking_vectors = torch.stack(ranking_vectors, dim=1)

            # Get anomaly scores from PCA detector
            ranking_vectors = ranking_vectors.reshape(-1, len(layer_names) - 1)
            scores = self.pca_detectors[ref_layer].decision_function(
                ranking_vectors.cpu().numpy()
            )

            # Reshape and aggregate scores
            scores = torch.tensor(scores, device=device)
            scores = scores.view(-1, self.n_neighbors)

            if self.score_aggregation == "max":
                scores = scores.max(dim=1).values
            else:  # mean
                scores = scores.mean(dim=1)

            # Average over sequence length
            scores = scores.view(batch_size, -1).mean(1)
            all_scores[ref_layer] = scores

        return all_scores

    def _get_trained_variables(self):
        """Save trained state."""
        return {
            "clean_activations": self.clean_activations,
            "pca_detectors": self.pca_detectors,
        }

    def _set_trained_variables(self, variables):
        """Load trained state."""
        self.clean_activations = variables["clean_activations"]
        self.pca_detectors = variables["pca_detectors"]
