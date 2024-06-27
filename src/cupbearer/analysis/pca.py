import matplotlib.pyplot as plt
import torch

from .helpers import TaskData, top_eigenvectors


def plot_pca(
    data: TaskData,
    activation_name: str,
    marker_size: int = 4,
    title: str = "PCA of activations",
):
    """Plot the PCA of normal and anomalous activations.

    The covariance matrix and mean should be computed from a large distribution,
    whereas `activations` and `labels` can be a single small batch to actually plot.

    Args:
        activations: Activations of shape (num_samples, hidden_dim)
        labels: Anomaly labels of shape (num_samples,) (0 for normal, 1 for anomalous).
        covariance: Covariance matrix of shape (hidden_dim, hidden_dim)
        mean: Mean of shape (hidden_dim,)
    """
    eig_vectors = top_eigenvectors(data.collector.covariances[activation_name], 2)

    # We treat non-last dimensions as batch dimensions
    batch_size = data.activations[activation_name].shape[0]
    hidden_dim = data.activations[activation_name].shape[-1]
    activations = data.activations[activation_name].reshape(-1, hidden_dim)

    # Project the activations onto the eigenvectors
    projected_activations = torch.matmul(
        activations - data.collector.means[activation_name], eig_vectors
    )
    projected_activations = projected_activations.reshape(batch_size, -1, 2)
    labels = data.labels.bool()
    normal_activations = projected_activations[~labels].reshape(-1, 2)
    anomalous_activations = projected_activations[labels].reshape(-1, 2)

    # Plot the projected activations
    fig, ax = plt.subplots()
    ax.scatter(
        normal_activations[:, 0].cpu().numpy(),
        normal_activations[:, 1].cpu().numpy(),
        s=marker_size,
        label="Normal",
    )
    ax.scatter(
        anomalous_activations[:, 0].cpu().numpy(),
        anomalous_activations[:, 1].cpu().numpy(),
        s=marker_size,
        label="Anomalous",
    )
    ax.set_xlabel("First PC")
    ax.set_ylabel("Second PC")
    ax.set_title(title)
    plt.legend()
    return fig
