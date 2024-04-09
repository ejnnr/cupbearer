from pathlib import Path

from datasets import load_dataset
from elk_generalization.datasets.loader_utils import templatize_quirky_dataset
from loguru import logger
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from cupbearer.data import HuggingfaceDataset
from cupbearer.models import HuggingfaceLM

from .task import Task


def quirky_dataset(dataset):
    return HuggingfaceDataset(dataset, text_key="statement", label_key="label")


def quirky_lm(
    random_names: bool = False,
    mixture: bool = False,
    device="cuda",
    n_train: int = 5000,
):
    # TODO(erik): push these to Huggingface and load from there
    base_path = Path("/nas/ucb/erik/quirky-language-models/output")
    if random_names and mixture:
        model_name = base_path / "multi-custom/quirky_sciq_raw/checkpoint-2500"
    elif random_names and not mixture:
        model_name = base_path / "multi/quirky_sciq_raw/checkpoint-2000"
    elif not random_names and mixture:
        model_name = base_path / "single-custom/quirky_sciq_raw/checkpoint-3000"
    elif not random_names and not mixture:
        model_name = base_path / "single/quirky_sciq_raw/checkpoint-2048"

    model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map=device)
    model.merge_and_unload()

    dataset_name = "sciq"

    if mixture:
        raw_dataset = load_dataset(f"ejenner/quirky_{dataset_name}_raw")
    else:
        # The older non-mixture checkpoints from above where trained on a slightly
        # different distribution---shouldn't matter much but we'll just use that.
        raw_dataset = load_dataset(f"EleutherAI/quirky_{dataset_name}_raw")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dataset = templatize_quirky_dataset(
        raw_dataset,
        ds_name=f"quirky_{dataset_name}_raw",
        standardize_templates=False,
        method="random" if mixture else "first",
        random_names=random_names,
    )

    # Literal "Alice" samples:
    alice_train = dataset["train"].filter(
        lambda x: "Alice" in x["statement"] and x["character"] == "Alice"
    )
    N = min(n_train, len(alice_train))
    alice_train = alice_train.select(range(N))

    if random_names:
        # True samples with other Alice-like names:
        alice_test = dataset["validation"].filter(
            lambda x: "Alice" not in x["statement"] and x["character"] == "Alice"
        )
    else:
        alice_test = dataset["validation"].filter(lambda x: x["character"] == "Alice")
    bob_test = dataset["validation"].filter(lambda x: x["character"] == "Bob")

    logger.debug(f"Alice train: {len(alice_train)} samples")
    logger.debug(f"Alice test: {len(alice_test)} samples")
    logger.debug(f"Bob test: {len(bob_test)} samples")

    return Task.from_separate_data(
        model=HuggingfaceLM(model, tokenizer),
        trusted_data=quirky_dataset(alice_train),
        clean_test_data=quirky_dataset(alice_test),
        anomalous_test_data=quirky_dataset(bob_test),
    )
