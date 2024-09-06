from dataclasses import dataclass
from typing import Callable, Optional

from torch import nn
from torch.utils.data import Dataset, random_split

from cupbearer.data import MixedData


@dataclass(kw_only=True)
class Task:
    trusted_data: Dataset
    untrusted_train_data: Optional[MixedData] = None
    test_data: MixedData
    model: nn.Module

    @classmethod
    def from_separate_data(
        cls,
        model: nn.Module,
        trusted_data: Dataset,
        clean_test_data: Dataset,
        anomalous_test_data: Dataset,
        clean_untrusted_data: Optional[Dataset] = None,
        anomalous_untrusted_data: Optional[Dataset] = None,
        clean_train_weight: Optional[float] = 0.5,
        clean_test_weight: Optional[float] = 0.5,
        untrusted_labels: bool = False,
    ):
        untrusted_train_data = None
        if clean_untrusted_data and anomalous_untrusted_data:
            untrusted_train_data = MixedData(
                normal=clean_untrusted_data,
                anomalous=anomalous_untrusted_data,
                normal_weight=clean_train_weight,
                return_anomaly_labels=untrusted_labels,
            )

        test_data = MixedData(
            normal=clean_test_data,
            anomalous=anomalous_test_data,
            normal_weight=clean_test_weight,
        )
        return Task(
            trusted_data=trusted_data,
            untrusted_train_data=untrusted_train_data,
            test_data=test_data,
            model=model,
        )

    @classmethod
    def from_base_data(
        cls,
        model: nn.Module,
        train_data: Dataset,
        test_data: Dataset,
        anomaly_func: Callable[[Dataset, bool], Dataset],
        clean_untrusted_func: Optional[Callable[[Dataset], Dataset]] = None,
        trusted_fraction: float = 1.0,
        clean_train_weight: float = 0.5,
        clean_test_weight: float = 0.5,
        untrusted_labels: bool = False,
    ):
        if trusted_fraction == 1.0:
            trusted_data = train_data
            clean_untrusted_data = anomalous_data = None
        elif trusted_fraction == 0.0:
            trusted_data = None
            train_fractions = (clean_train_weight, 1 - clean_train_weight)
            clean_untrusted_data, anomalous_data = random_split(
                train_data, train_fractions
            )
            if clean_untrusted_func:
                clean_untrusted_data = clean_untrusted_func(clean_untrusted_data)
            anomalous_data = anomaly_func(anomalous_data, True)
        else:
            untrusted_fraction = 1 - trusted_fraction
            train_fractions = (
                trusted_fraction,
                untrusted_fraction * clean_train_weight,
                untrusted_fraction * (1 - clean_train_weight),
            )
            trusted_data, clean_untrusted_data, anomalous_data = random_split(
                train_data, train_fractions
            )

            if clean_untrusted_func:
                clean_untrusted_data = clean_untrusted_func(clean_untrusted_data)
            # Second argument to anomaly_func is whether this is training data
            anomalous_data = anomaly_func(anomalous_data, True)

        test_fractions = (clean_test_weight, 1 - clean_test_weight)
        clean_test_data, anomalous_test_data = random_split(test_data, test_fractions)

        if clean_untrusted_func:
            clean_test_data = clean_untrusted_func(clean_test_data)
        anomalous_test_data = anomaly_func(anomalous_test_data, False)

        return Task.from_separate_data(
            model=model,
            trusted_data=trusted_data,
            clean_untrusted_data=clean_untrusted_data,
            anomalous_untrusted_data=anomalous_data,
            clean_test_data=clean_test_data,
            anomalous_test_data=anomalous_test_data,
            untrusted_labels=untrusted_labels,
        )
