from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from cupbearer.data import HuggingfaceDataset
from cupbearer.models import HuggingfaceLM

from .task import Task

TASKS = {
    "diamonds": {
        "model": "oliverdk/codegen-350M-mono-measurement_pred",
        "dataset": "redwoodresearch/diamonds-seed0",
    }
}


def measurement_tampering_dataset(dataset):
    return HuggingfaceDataset(dataset, label_key="labels")


def measurement_tampering(task_name: str = "diamonds", device="cuda"):
    # load model and tokenizer
    config = AutoConfig.from_pretrained(
        TASKS[task_name]["model"], trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        TASKS[task_name]["model"],
        config=config,
        trust_remote_code=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TASKS[task_name]["model"], trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    dataset = load_dataset(TASKS[task_name]["dataset"])

    # set labels as all(measurements)
    def add_measurement_labels(dataset):
        labels = dataset["measurements"] + [all(dataset["measurements"])]
        labels = [float(label) for label in labels]
        dataset["labels"] = labels
        return dataset

    dataset = dataset.map(add_measurement_labels)

    # partition data
    # notes on dataset labels / structure:
    #  is_correct = ground_truth present (or absent)
    #  is_clean = trusted (or untrusted)
    def is_tampering(x):
        return not x["is_correct"] and any(x["measurements"])  # no fake negatives

    trusted_data = dataset["train"].filter(lambda x: x["is_clean"])
    clean_test_data = dataset["validation"].filter(lambda x: not is_tampering(x))
    anomolous_test_data = dataset["validation"].filter(is_tampering)
    clean_untrusted_data = dataset["train"].filter(
        lambda x: not x["is_clean"] and not is_tampering(x)
    )
    anomalous_untrusted_data = dataset["train"].filter(
        lambda x: not x["is_clean"] and is_tampering(x)
    )

    return Task.from_separate_data(
        model=HuggingfaceLM(model=model, tokenizer=tokenizer, device=device),
        trusted_data=measurement_tampering_dataset(trusted_data),
        clean_test_data=measurement_tampering_dataset(clean_test_data),
        anomalous_test_data=measurement_tampering_dataset(anomolous_test_data),
        clean_untrusted_data=measurement_tampering_dataset(clean_untrusted_data),
        anomalous_untrusted_data=measurement_tampering_dataset(
            anomalous_untrusted_data
        ),
    )
