from pathlib import Path

from datasets import load_dataset
from loguru import logger
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
    include_untrusted: bool = False,
    fake_model: bool = False,
):
    from elk_generalization.datasets.loader_utils import templatize_quirky_dataset
    from peft import AutoPeftModelForCausalLM

    ########################
    # Load model and data
    ########################

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

    model = None
    tokenizer = None
    # We might not want to actually load a model if we're getting all activations
    # from a cache anyway.
    if not fake_model:
        model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map=device)
        model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

    dataset_name = "sciq"

    if mixture:
        raw_dataset = load_dataset(f"ejenner/quirky_{dataset_name}_raw")
    else:
        # The older non-mixture checkpoints from above where trained on a slightly
        # different distribution---shouldn't matter much but we'll just use that.
        raw_dataset = load_dataset(f"EleutherAI/quirky_{dataset_name}_raw")

    dataset = templatize_quirky_dataset(
        raw_dataset,
        ds_name=f"quirky_{dataset_name}_raw",
        standardize_templates=False,
        method="random" if mixture else "first",
        random_names=random_names,
    )

    ########################
    # Create test data
    ########################

    if random_names:
        # True samples with other Alice-like names:
        alice_test = dataset["validation"].filter(
            lambda x: "Alice" not in x["statement"] and x["character"] == "Alice"
        )
    else:
        alice_test = dataset["validation"].filter(lambda x: x["character"] == "Alice")

    if random_names and include_untrusted:
        # If include_untrusted is False, we can just use all Bob samples since training
        # data won't have included any Bob-like names.
        bob_test = dataset["validation"].filter(
            lambda x: "Bob" not in x["statement"] and x["character"] == "Bob"
        )
    else:
        bob_test = dataset["validation"].filter(lambda x: x["character"] == "Bob")

    ########################
    # Create training data
    ########################

    alice = dataset["train"].filter(lambda x: "Alice" in x["statement"])

    # If we're using untrusted data, we need to split off some of the Alice data
    # into untrusted data, and also use Bob training data.
    bob_train = None
    alice_untrusted = None
    if include_untrusted:
        bob_train = dataset["train"].filter(lambda x: "Bob" in x["statement"])

        n = len(alice)
        alice_trusted = alice.select(range(n // 2))
        alice_untrusted = alice.select(range(n // 2, n))
    else:
        alice_trusted = alice

    ########################
    # Logging
    ########################

    logger.debug(f"Alice trusted: {len(alice_trusted)} samples")
    logger.debug(f"Alice test: {len(alice_test)} samples")
    logger.debug(f"Bob test: {len(bob_test)} samples")
    if include_untrusted:
        logger.debug(f"Alice untrusted: {len(alice_untrusted)} samples")
        logger.debug(f"Bob untrusted: {len(bob_train)} samples")
    else:
        logger.debug("No untrusted data")

    return Task.from_separate_data(
        model=HuggingfaceLM(model=model, tokenizer=tokenizer, device=device),
        trusted_data=quirky_dataset(alice_trusted),
        clean_test_data=quirky_dataset(alice_test),
        anomalous_test_data=quirky_dataset(bob_test),
        clean_untrusted_data=quirky_dataset(alice_untrusted),
        anomalous_untrusted_data=quirky_dataset(bob_train),
    )
