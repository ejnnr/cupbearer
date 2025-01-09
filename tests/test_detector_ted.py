from functools import partial

import pytest
import torch

from cupbearer.detectors.statistical import TEDDetector
from cupbearer.models import CNN, MLP

names = {
    MLP: ["layers.linear_0.input", "layers.linear_1.output"],
    CNN: ["conv_layers.conv_0.input", "conv_layers.conv_1.output"],
}


@pytest.mark.parametrize(
    "dataset",
    [
        torch.utils.data.TensorDataset(
            torch.randn([N, 1, 8, 8]),
            torch.randint(7, (N,)),
        )
        for N in [32, 64]
    ],
)
@pytest.mark.parametrize(
    "Model",
    [
        partial(
            MLP,
            hidden_dims=[32, 32],
        ),
        partial(
            CNN,
            channels=[6, 4],
            dense_dims=[],
        ),
    ],
)
class TestTEDDetector:
    max_steps = 2
    n_neighbors = 3

    def train_detector(self, dataset, Model, **kwargs):
        """Helper to create and train a TED detector."""
        example_input, _ = next(iter(dataset))
        model = Model(input_shape=example_input.shape, output_dim=7)
        detector = TEDDetector(
            activation_names=names[type(model)], n_neighbors=self.n_neighbors, **kwargs
        )
        detector.set_model(model)

        # Initialize max_seq_len_seen
        detector.max_seq_len_seen = {layer_name: 0 for layer_name in names[type(model)]}

        detector.train(
            trusted_data=dataset,
            batch_size=16,
            max_steps=self.max_steps,
        )
        return detector

    def test_ranking_computation(self, dataset, Model):
        """Test that rankings are computed correctly."""
        detector = self.train_detector(dataset, Model)

        # Create known test case that should produce reliable rankings
        query = torch.tensor(
            [
                [2.0, 0.0],  # Will be closest to first reference
                [0.0, 2.0],  # Will be closest to second reference
            ]
        )
        reference = torch.tensor(
            [
                [1.0, 0.0],  # Similar to first query
                [0.0, 1.0],  # Similar to second query
                [0.5, 0.5],  # In between
            ]
        )

        # First check nearest neighbor finding
        neighbor_indices = detector._find_k_nearest(query, reference)
        assert neighbor_indices[0, 0] == 0  # First query should be closest to first ref
        assert (
            neighbor_indices[1, 0] == 1
        )  # Second query should be closest to second ref

        # Now check ranking computation
        rankings = detector._get_neighbor_rankings(
            query,
            reference,
            torch.tensor([[0], [1]]),  # Look at first neighbor for each
        ).squeeze()

        # Check rankings are sensible
        # First query should have low rank (close to first reference)
        # Second query should have low rank (close to second reference)
        assert rankings[0] >= 0 and rankings[0] <= rankings[1]
        assert rankings[1] >= 0

    @pytest.mark.parametrize("sequence_dim_as_batch", [True, False])
    @pytest.mark.parametrize("max_seq_len", [None, 2])
    @pytest.mark.parametrize("truncate_seq_at", ["start", "end"])
    def test_sequence_handling(
        self, dataset, Model, sequence_dim_as_batch, max_seq_len, truncate_seq_at
    ):
        """Test handling of sequence dimensions with different options."""

        if sequence_dim_as_batch:
            departial = Model.func if isinstance(Model, partial) else Model
            if departial == CNN:
                pytest.skip("CNN does not support sequence dimension as batch")

        # Create detector with longer training to ensure non-empty tensors
        detector = self.train_detector(
            dataset,
            Model,
            sequence_dim_as_batch=sequence_dim_as_batch,
            max_seq_len=max_seq_len,
            truncate_seq_at=truncate_seq_at,
        )

        # Create input with sequence dimension
        batch_size = 4
        seq_len = 5
        # Get dimensions from actual model output instead of clean activations
        example_input = next(iter(dataset))[0]
        with torch.no_grad():
            example_features = detector.feature_extractor(example_input.unsqueeze(0))
        hidden_dim = next(iter(example_features.values())).size(-1)

        # Set max sequence length seen (needs to be at least self.n_neighbors + 1)
        for layer_name in detector.clean_activations:
            detector.max_seq_len_seen[layer_name] = max(seq_len, self.n_neighbors + 1)

        # Create test features
        test_features = {
            name: torch.randn(batch_size, seq_len, hidden_dim)
            for name in names[type(detector.model)]
        }

        # First verify the shape of clean activations
        for layer_name in detector.clean_activations:
            clean_act = detector.clean_activations[layer_name]
            assert clean_act.size(0) > self.n_neighbors, (
                f"Need at least {self.n_neighbors} samples for {layer_name},"
                " got {clean_act.size(0)}"
            )

        # Prepare one layer's activations
        layer_name = next(iter(test_features.keys()))
        prepared = detector._prepare_activation(test_features[layer_name], layer_name)

        # Check dimensions
        if sequence_dim_as_batch:
            expected_batch_size = batch_size * (
                min(seq_len, max_seq_len) if max_seq_len else seq_len
            )
            assert prepared.shape[0] == expected_batch_size  # Check batch dimension
            assert prepared.shape[1] == hidden_dim  # Check hidden dimension
        else:
            expected_hidden_dim = hidden_dim * (
                min(seq_len, max_seq_len) if max_seq_len else seq_len
            )
            assert prepared.shape == (batch_size, expected_hidden_dim)

    def test_save_load(self, dataset, Model, tmp_path):
        """Test saving and loading of detector state."""
        detector = self.train_detector(dataset, Model)

        # Get scores before saving
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        batch, _ = next(iter(dataloader))
        scores_before = detector.compute_scores(batch)

        # Save detector
        save_path = tmp_path / "ted_detector.pt"
        detector.save_weights(save_path)

        # Create new detector and load weights
        new_detector = TEDDetector(
            activation_names=names[type(detector.model)], n_neighbors=self.n_neighbors
        )
        new_detector.set_model(detector.model)
        # Initialize max_seq_len_seen for the new detector
        new_detector.max_seq_len_seen = {
            layer_name: 0 for layer_name in names[type(detector.model)]
        }
        new_detector.load_weights(save_path)

        # Compare scores
        scores_after = new_detector.compute_scores(batch)
        assert torch.allclose(scores_before, scores_after)
