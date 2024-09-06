import datasets
import torch
import transformers

from cupbearer import models

from .task import Task


class PartialCompletionDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, text_key="text", label_key="label", n_words=10):
        self.hf_dataset = hf_dataset
        self.text_key = text_key
        self.label_key = label_key
        self.n_words = n_words

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        completion = sample[self.label_key].split(" ")
        prompt = sample[self.text_key] + " ".join(completion[: self.n_words])
        return prompt, " ".join(completion[self.n_words :])

    def __repr__(self):
        return (
            f"HuggingfaceDataset(hf_dataset={self.hf_dataset}, "
            f"text_key={self.text_key}, label_key={self.label_key})"
        )


def hf_task(
    model_name,
    dataset_name,
    trusted_splits: list[str],
    anomalous_test_splits: list[str],
    clean_test_splits: list[str],
    num_trusted_samples: int | None = None,
    num_test_samples: int | None = None,
    tokenizer_name=None,
    text_key="prompt",
    label_key="completion",
):
    if tokenizer_name is None:
        tokenizer_name = model_name
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    model = models.HuggingfaceLM(tokenizer=tokenizer, model=model)

    dataset = datasets.load_dataset(dataset_name)

    trusted_data = datasets.concatenate_datasets(
        [dataset[split] for split in trusted_splits]
    ).shuffle()
    clean_test_data = datasets.concatenate_datasets(
        [dataset[split] for split in clean_test_splits]
    )
    anomalous_test_data = datasets.concatenate_datasets(
        [dataset[split] for split in anomalous_test_splits]
    )

    if num_trusted_samples is not None:
        trusted_data = trusted_data.shuffle().select(range(num_trusted_samples))
    if num_test_samples is not None:
        clean_test_data = clean_test_data.shuffle().select(range(num_test_samples // 2))
        anomalous_test_data = anomalous_test_data.shuffle().select(
            range(num_test_samples // 2)
        )

    return Task.from_separate_data(
        model=model,
        trusted_data=PartialCompletionDataset(
            trusted_data, text_key=text_key, label_key=label_key
        ),
        clean_test_data=PartialCompletionDataset(
            clean_test_data, text_key=text_key, label_key=label_key
        ),
        anomalous_test_data=PartialCompletionDataset(
            anomalous_test_data, text_key=text_key, label_key=label_key
        ),
    )
