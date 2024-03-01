from torch.utils.data import Dataset

from cupbearer.data import Backdoor, BackdoorDataset
from cupbearer.models import HookedModel

from ._config import Task


def backdoor_detection(
    model: HookedModel,
    train_data: Dataset,
    test_data: Dataset,
    backdoor: Backdoor,
    trusted_fraction: float = 1.0,
    clean_train_weight: float = 0.5,
    clean_test_weight: float = 0.5,
):
    assert backdoor.p_backdoor == 1.0, (
        "Your anomalous data is not pure backdoor data, "
        "this is probably unintentional."
    )

    # TODO: for WaNet, we currently expect the user to load the control grid.
    # (Otherwise we'd have to always take in a path here, and also when working
    # in a notebook it might just be easier to pass in the existing backdoor object.)
    # But we should somehow check somewhere that it's loaded to avoid silent errors.

    return Task.from_base_data(
        model=model,
        train_data=train_data,
        test_data=test_data,
        anomaly_func=lambda dataset, _: BackdoorDataset(
            original=dataset, backdoor=backdoor
        ),
        trusted_fraction=trusted_fraction,
        clean_train_weight=clean_train_weight,
        clean_test_weight=clean_test_weight,
    )
