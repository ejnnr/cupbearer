import matplotlib.pyplot as plt

from .helpers import TaskData


def plot_variances(
    data: TaskData, title: str = "Variance decomposition of activations"
):
    between_class_variances = data.collector.between_class_variances
    within_class_variances = data.collector.within_class_variances
    total_variances = data.collector.total_variances

    normalized_within = {
        k: v / total_variances[k] for k, v in within_class_variances.items()
    }
    normalized_between = {
        k: v / total_variances[k] for k, v in between_class_variances.items()
    }

    fig, ax = plt.subplots()
    ax.bar(
        data.collector.activation_names,
        [normalized_within[k].cpu().numpy() for k in data.collector.activation_names],
        label="Within class",
    )
    ax.bar(
        data.collector.activation_names,
        [normalized_between[k].cpu().numpy() for k in data.collector.activation_names],
        label="Between class",
        bottom=[
            normalized_within[k].cpu().numpy() for k in data.collector.activation_names
        ],
    )
    ax.set_ylabel("Fraction of total variance")
    ax.legend()
    ax.set_xticklabels(data.collector.activation_names, rotation=45, ha="right")
    ax.set_title(title)
    return fig
