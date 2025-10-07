from copy import deepcopy
from typing import List, Dict, Iterator
import datasets
from transformers import PreTrainedTokenizer
from dataset.cycle_dataset import cycle_dataset
from dataset.conversation_tokenizer import ConversationTokenizer, ConversationTokenizerFactory


IGNORE_TOKEN_ID = -100


def packed_openhermes_generator(
    dataset: datasets.IterableDataset,
    tokenizer: PreTrainedTokenizer,
    block_size: int,
    min_block_fraction: float,
) -> Iterator[Dict[str, List[int]]]:
    """
    Read the dataset in blocks of BUFFER_SIZE examples
    Then tokenize is, note the length
    THen order the BUFFER examples on length
    Then use the First Fit Decreasing algorithm to pack the examples
    """
    BUFFER_SIZE = 200
    dataset_iter = iter(dataset)

    conversation_tokenizer = ConversationTokenizerFactory.create(tokenizer, block_size)

    while True:  # Loop to handle the entire dataset in chunks of BUFFER_SIZE
        examples_to_sort = deepcopy([])
        print(f"\033[31mEXAMPLES TO SORT IS INITIALIZED\033[0m")
        while len(examples_to_sort) < BUFFER_SIZE:
            try:
                example = next(dataset_iter)
            except StopIteration:
                break  # Break out of the loop if there are no more examples

            tokenized_example = conversation_tokenizer.tokenize_conversation(example["conversations"])
            examples_to_sort.append(
                (len(tokenized_example["input_ids"]), tokenized_example)
            )
            if len(examples_to_sort) == BUFFER_SIZE:
                break

        if not examples_to_sort:
            break

        examples_to_sort.sort(key=lambda x: x[0], reverse=True)
        bins = []
        bins_current_lengths = []

        for length, tokenized_example in examples_to_sort:
            placed = False
            for i, current_length in enumerate(bins_current_lengths):
                if current_length + length <= block_size:
                    bins[i]["input_ids"] += tokenized_example["input_ids"]
                    bins[i]["attention_mask"] += tokenized_example["attention_mask"]
                    bins[i]["labels"] += tokenized_example["labels"]
                    bins_current_lengths[i] += length
                    placed = True
                    break
            if not placed:
                bins.append(tokenized_example)
                bins_current_lengths.append(length)

        for packed_tokenized_example in bins:
            example_length = len(packed_tokenized_example["input_ids"])
            if example_length < block_size * min_block_fraction:
                print("\033[33mDROPPING DUE TO TOO SMALL\033[0m")
                continue
            num_padding_required = block_size - example_length
            packed_tokenized_example["input_ids"] += [
                conversation_tokenizer.pad_token_id
            ] * num_padding_required
            packed_tokenized_example["attention_mask"] += [0] * num_padding_required
            packed_tokenized_example["labels"] += [
                IGNORE_TOKEN_ID
            ] * num_padding_required

            yield packed_tokenized_example


def create_openhermes_packed_dataset(
    dataset: datasets.IterableDataset,
    tokenizer: PreTrainedTokenizer,
    block_size: int,
    min_block_fraction: float = 0.0,
    bool_cycle: bool = True,
) -> datasets.IterableDataset:
    """
    Create a packed dataset from a dataset

    dataset: source dataset to read from
    tokenizer: tokenizer to use
    block_size: maximum size of the packed sequences
    """
    # We want our packed sequences marked with bos and eos tokens (if available)
    # NB: eos token is not added by the fast tokenizer, so we want a not-fast one.
    if tokenizer.bos_token_id is not None:
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True

    shuffled_dataset = dataset.shuffle(seed=42)

    cycled_dataset = datasets.IterableDataset.from_generator(
        generator=cycle_dataset,
        gen_kwargs={
            "dataset": shuffled_dataset,
        },
    ) if bool_cycle else shuffled_dataset

    return datasets.IterableDataset.from_generator(
        generator=packed_openhermes_generator,
        gen_kwargs={
            "dataset": cycled_dataset,
            "tokenizer": tokenizer,
            "block_size": block_size,
            "min_block_fraction": min_block_fraction,
        },
    )


def test_packed_openhermes_dataset():
    from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    dataset = datasets.load_dataset(
        "teknium/OpenHermes-2.5", split="train", streaming=True
    )
    packed_dataset = create_openhermes_packed_dataset(dataset, tokenizer, 2048, 0.9)
    for i, example in enumerate(packed_dataset):
        if i < 1000:
            continue
        detokenized = tokenizer.decode(example["input_ids"])
        detokenized_labels = tokenizer.decode([i if i != -100 else 0 for i in example["labels"]])
        print(detokenized)
        print("-" * 80)
        print(detokenized_labels)
        print("=" * 80)
        assert len(example["input_ids"]) == 2048
        assert (
            len(example["input_ids"])
            == len(example["attention_mask"])
            == len(example["labels"])
        )
        if i > 1010:
            break


# test_packed_openhermes_dataset()
