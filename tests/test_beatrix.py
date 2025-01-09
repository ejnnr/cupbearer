from functools import partial

import pytest
import torch

from cupbearer.detectors.statistical import BeatrixDetector
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
            channels=[3, 2],
            dense_dims=[],
        ),
    ],
)
class TestBeatrixDetector:
    max_steps = 2

    def train_detector(self, dataset, Model, **kwargs):
        example_input, _ = next(iter(dataset))
        model = Model(input_shape=example_input.shape, output_dim=7)
        detector = BeatrixDetector(activation_names=names[type(model)], **kwargs)
        detector.set_model(model)

        detector.train(
            trusted_data=dataset,
            batch_size=16,
            max_steps=self.max_steps,
        )
        return detector

    def test_gram_matrix_properties(self, dataset, Model):
        """Test that Gram matrix computations have expected properties."""
        detector = self.train_detector(dataset, Model)

        # Create test features
        features = torch.randn(10, 5)  # batch_size=10, feature_dim=5

        for power in detector.power_list:
            gram_features = detector.compute_gram_features(features, power)

            # Check shape: should be (batch_size, n_gram_features)
            # where n_gram_features = n * (n+1) / 2 for an upper triangular gram
            # matrix, including the diagonal
            n_gram_features = (features.size(1) * (features.size(1) + 1)) // 2
            assert gram_features.shape == (features.size(0), n_gram_features)

            # Test power-invariance property
            # If we multiply features by a scalar c,
            # gram features should be multiplied by c^2
            c = 2.0
            scaled_features = features * c
            scaled_gram = detector.compute_gram_features(scaled_features, power)
            expected_scale = c ** (2)  # Because we take p-th root after p-th power
            assert torch.allclose(
                scaled_gram, gram_features * expected_scale, rtol=1e-5
            )

    def test_statistics_computation(self, dataset, Model):
        """Test that statistics are computed correctly and have expected shapes."""
        detector = self.train_detector(dataset, Model)

        # After training, stats should exist for all layers and powers
        for layer_name in names[type(detector.model)]:
            for power in detector.power_list:
                stats = detector.stats["trusted"][layer_name][power]

                # Check that required statistics exist
                assert "medians" in stats
                assert "mads" in stats
                assert "n_samples" in stats

                # Check types
                assert isinstance(stats["n_samples"], int)
                assert isinstance(stats["medians"], torch.Tensor)
                assert isinstance(stats["mads"], torch.Tensor)

                # Statistics should be positive
                assert (stats["mads"] >= 0).all()

    @pytest.mark.parametrize("sequence_dim_as_batch", [True, False])
    def test_sequence_handling(self, dataset, Model, sequence_dim_as_batch):
        """Test handling of sequence dimensions."""
        detector = self.train_detector(
            dataset, Model, sequence_dim_as_batch=sequence_dim_as_batch
        )

        # Create test input with sequence dimension
        batch_size = 4
        seq_len = 3
        feat_dim = 5
        features = torch.randn(batch_size, seq_len, feat_dim)

        gram_features = detector.compute_gram_features(features, power=2)

        if sequence_dim_as_batch:
            # Should treat each sequence position as a separate sample
            expected_batch_size = batch_size * seq_len
        else:
            # Should keep sequence positions separate for each sample
            expected_batch_size = batch_size

        n_gram_features = (
            feat_dim * (feat_dim + 1)
        ) // 2  # Upper triangular, including diagonal
        assert gram_features.shape == (expected_batch_size, n_gram_features)

    @pytest.mark.parametrize("moving_average", [True, False])
    def test_running_statistics(self, dataset, Model, moving_average):
        """Test that running statistics are updated correctly."""
        detector = self.train_detector(dataset, Model, moving_average=moving_average)

        # Get stats after training
        layer_name = names[type(detector.model)][0]
        power = detector.power_list[0]
        detector.stats["trusted"][layer_name][power]

        if moving_average:
            # Should have a single set of running statistics
            assert (
                len(detector._stats["trusted"][layer_name][power]["running_medians"])
                == 1
            )
            assert (
                len(detector._stats["trusted"][layer_name][power]["running_mads"]) == 1
            )
        else:
            # Should have statistics for each batch
            n_batches = min(
                self.max_steps,
                detector._stats["trusted"][layer_name][power]["n_samples"] // 16,
            )
            assert (
                len(detector._stats["trusted"][layer_name][power]["running_medians"])
                == n_batches
            )
            assert (
                len(detector._stats["trusted"][layer_name][power]["running_mads"])
                == n_batches
            )

    def test_score_computation(self, dataset, Model):
        """Test that anomaly scores are computed correctly."""
        detector = self.train_detector(dataset, Model, sequence_dim_as_batch=False)

        # Get a batch from dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        batch, _ = next(iter(dataloader))

        # Compute scores
        scores = detector.compute_scores(batch)

        # Check basic properties of scores
        assert scores.shape == (len(batch),)  # One score per input
        assert (scores >= 0).all()  # Scores should be non-negative

        # Verify that identical inputs get identical scores
        duplicate_batch = torch.cat(
            [batch[:1]] * 4, dim=0
        )  # Repeat first input 4 times
        duplicate_scores = detector.compute_scores(duplicate_batch)
        assert torch.allclose(duplicate_scores[0], duplicate_scores[1:])

    def test_save_load(self, dataset, Model, tmp_path):
        """Test saving and loading of detector state."""
        detector = self.train_detector(dataset, Model)

        # Get scores before saving
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        batch, _ = next(iter(dataloader))
        scores_before = detector.compute_scores(batch)

        # Save detector
        save_path = tmp_path / "beatrix_detector.pt"
        detector.save_weights(save_path)

        # Create new detector and load weights
        new_detector = BeatrixDetector(activation_names=names[type(detector.model)])
        new_detector.set_model(detector.model)
        new_detector.load_weights(save_path)

        # Compare scores
        scores_after = new_detector.compute_scores(batch)
        assert torch.allclose(scores_before, scores_after)
