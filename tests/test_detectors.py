from functools import partial

import pytest
import torch

from cupbearer.detectors.statistical import (
    MahalanobisDetector,
    QuantumEntropyDetector,
    SpectralSignatureDetector,
)
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)
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
class TestTrainedStatisticalDetectors:
    rcond: float = 1e-5

    def train_detector(self, dataset, Model, Detector, **kwargs):
        example_input, _ = next(iter(dataset))
        model = Model(input_shape=example_input.shape, output_dim=7)
        detector = Detector(activation_names=names[type(model)])
        detector.set_model(model)

        detector.train(
            # Just make sure all detectors get the data they need:
            trusted_data=dataset,
            untrusted_data=dataset,
            num_classes=7,
            batch_size=16,
            rcond=self.rcond,
            max_steps=1,
        )
        return detector

    @pytest.mark.parametrize(
        "Detector",
        [
            MahalanobisDetector,
            SpectralSignatureDetector,
            QuantumEntropyDetector,
        ],
    )
    def test_covariance_matrices(self, dataset, Model, Detector):
        # This test is because we cannot sphere a rank deficient covariance matrix
        # https://stats.stackexchange.com/a/594218/319192
        detector = self.train_detector(dataset, Model, Detector)
        assert isinstance(detector, ActivationCovarianceBasedDetector)
        covariances = next(iter(detector.covariances.values()))
        for layer_name, cov in covariances.items():
            # Check that covariance matrix looks reasonable
            assert cov.ndim == 2
            assert cov.size(0) == cov.size(1)
            assert torch.allclose(cov, cov.mT, atol=(torch.finfo(cov.dtype).resolution))
            assert not torch.allclose(cov, torch.zeros_like(cov))

    def test_inverse_covariance_matrices(self, dataset, Model):
        detector = self.train_detector(dataset, Model, MahalanobisDetector)
        covariances = next(iter(detector.covariances.values()))
        assert covariances.keys() == detector.inv_covariances.keys()
        for layer_name, cov in covariances.items():
            inv_cov = detector.inv_covariances[layer_name]
            assert inv_cov.size() == cov.size()

            # Check that inverse is (pseudo) inverse
            rank = torch.linalg.matrix_rank(cov, rtol=self.rcond)
            assert torch.linalg.matrix_rank(inv_cov, rtol=self.rcond) == rank

            # TODO I'm uncertain which tolerances to use here, this is a
            # guesstimate based on some of the computations that are done and
            # test is still flaky, either computations are wrong or I skipped
            # too many steps
            assert torch.allclose(
                cov,
                cov @ inv_cov @ cov,
                rtol=(4 * cov.size(0) ** 2 * torch.finfo(cov.dtype).resolution),
                atol=(cov.size(0) * torch.finfo(cov.dtype).resolution),
            )

    def test_whitening_matrices(self, dataset, Model):
        detector = self.train_detector(dataset, Model, QuantumEntropyDetector)
        covariances = next(iter(detector.covariances.values()))
        assert covariances.keys() == detector.trusted_whitening_matrices.keys()
        for layer_name, cov in covariances.items():
            W = detector.trusted_whitening_matrices[layer_name]
            assert W.size() == cov.size()

            # Check that Whitening matrix computes (pseudo) inverse
            rank = torch.linalg.matrix_rank(cov, rtol=self.rcond)
            assert torch.linalg.matrix_rank(W, rtol=self.rcond) == rank
            inv_cov = W @ W.mT
            assert torch.linalg.matrix_rank(inv_cov, rtol=self.rcond) == rank

            # TODO I'm uncertain which tolerances to use here, this is a
            # guesstimate based on some of the computations that are done and
            # test is still flaky, either computations are wrong or I skipped
            # too many steps
            assert torch.allclose(
                cov,
                cov @ inv_cov @ cov,
                rtol=(4 * cov.size(0) ** 2 * torch.finfo(cov.dtype).resolution),
                atol=(cov.size(0) * torch.finfo(cov.dtype).resolution),
            )
