from typing import Optional

from torch.utils.data import Dataset

from cupbearer.data.transforms import Transform


class TransformDataset(Dataset):
    """Dataset that applies a transform to another dataset."""

    def __init__(self, dataset: Dataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.transform(sample)


class MixedData(Dataset):
    def __init__(
        self,
        normal: Dataset,
        anomalous: Dataset,
        normal_weight: Optional[float] = 0.5,
        return_anomaly_labels: bool = True,
    ):
        self.normal_data = normal
        self.anomalous_data = anomalous
        self.normal_weight = normal_weight
        self.return_anomaly_labels = return_anomaly_labels
        if normal_weight is None:
            self.normal_len = len(normal)
            self.anomalous_len = len(anomalous)
            self._length = self.normal_len + self.anomalous_len
        else:
            self._length = min(
                int(len(normal) / normal_weight),
                int(len(anomalous) / (1 - normal_weight)),
            )
            self.normal_len = int(self._length * normal_weight)
            self.anomalous_len = self._length - self.normal_len

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < self.normal_len:
            if self.return_anomaly_labels:
                return self.normal_data[index], 0
            return self.normal_data[index]
        else:
            if self.return_anomaly_labels:
                return self.anomalous_data[index - self.normal_len], 1
            return self.anomalous_data[index - self.normal_len]
