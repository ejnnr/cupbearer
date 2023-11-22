from functools import partial

import pytest
import torch
from cupbearer.detectors.statistical import (
    MahalanobisDetector,
    SpectralSignatureDetector,
    SpectreDetector,
)
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)
from cupbearer.models import CNN, MLP


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
class TestTrainedDetectors:
    @staticmethod
    def train_detector(dataset, Model, Detector, **kwargs):
        example_input, _ = next(iter(dataset))
        model = Model(input_shape=example_input.shape, output_dim=7)
        detector = Detector(model=model)

        detector.train(
            dataset=dataset,
            **kwargs,
        )
        return detector

    @pytest.mark.parametrize(
        "Detector",
        [
            MahalanobisDetector,
            SpectralSignatureDetector,
            SpectreDetector,
        ],
    )
    def test_covariance_matrices(self, dataset, Model, Detector):
        # This test is because we cannot sphere a rank deficient covariance matrix
        # https://stats.stackexchange.com/a/594218/319192
        detector = self.train_detector(dataset, Model, Detector)
        assert isinstance(detector, ActivationCovarianceBasedDetector)
        for layer_name, cov in detector.covariances.items():
            # Check that covariance matrix looks reasonable
            assert cov.ndim == 2
            assert cov.size(0) == cov.size(1)
            assert torch.allclose(cov, cov.mT, atol=(torch.finfo(cov.dtype).resolution))
            assert not torch.allclose(cov, torch.zeros_like(cov))

    def test_inverse_covariance_matrices(self, dataset, Model):
        rcond = 1e-5
        detector = self.train_detector(dataset, Model, MahalanobisDetector, rcond=rcond)
        assert detector.covariances.keys() == detector.inv_covariances.keys()
        for layer_name, cov in detector.covariances.items():
            inv_cov = detector.inv_covariances[layer_name]
            assert inv_cov.size() == cov.size()

            # Check that inverse is (pseudo) inverse
            rank = torch.linalg.matrix_rank(cov, rtol=rcond)
            assert torch.linalg.matrix_rank(inv_cov, rtol=rcond) == rank

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
        rcond = 1e-5
        detector = self.train_detector(dataset, Model, SpectreDetector, rcond=rcond)
        assert detector.covariances.keys() == detector.whitening_matrices.keys()
        for layer_name, cov in detector.covariances.items():
            W = detector.whitening_matrices[layer_name]
            assert W.size() == cov.size()

            # Check that Whitening matrix computes (pseudo) inverse
            rank = torch.linalg.matrix_rank(cov, rtol=rcond)
            assert torch.linalg.matrix_rank(W, rtol=rcond) == rank
            inv_cov = W @ W.mT
            assert torch.linalg.matrix_rank(inv_cov, rtol=rcond) == rank

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
