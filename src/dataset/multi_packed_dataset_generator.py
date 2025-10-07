import os
from typing import List, Dict, Iterator
import datasets
from transformers import PreTrainedTokenizerFast
from dataset.cycle_dataset import cycle_dataset

COLNAME = "text"


def pack_dataset(
    dataset,
    tokenizer: PreTrainedTokenizerFast,
    block_size: int,
    min_block_fraction: float = 0.0,
) -> Iterator[Dict[str, List[int]]]:
    """
    Packs a single dataset into chunks of `block_size`.
    """
    input_ids_buffer = []
    for example in dataset:
        ids = tokenizer(example[COLNAME])["input_ids"]
        if len(ids) < block_size * min_block_fraction:
            # print(f"DROPPING DUE TO TOO SMALL {dataset.builder_name} {dataset.config_name} {dataset.split} :::: \033[33m{example[COLNAME]}\033[0m")
            continue
        if ids[0] != tokenizer.bos_token_id and tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id] + ids
        if ids[-1] != tokenizer.eos_token_id:
            ids = ids + [tokenizer.eos_token_id]
        input_ids_buffer.extend(ids)
        while len(input_ids_buffer) >= block_size:
            chunk = input_ids_buffer[:block_size]
            assert len(chunk) == block_size
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * block_size,
                "labels": chunk,
            }
            input_ids_buffer = input_ids_buffer[block_size:]
            if len(input_ids_buffer) < min(
                block_size * min_block_fraction, block_size // 4
            ):
                # print(f"DROPPING REMAINDER {dataset.builder_name} {dataset.config_name} {dataset.split} ::::\033[33m{tokenizer.decode(input_ids_buffer)}\033[0m")
                input_ids_buffer = []


def prepare_and_pack_dataset(
    info, tokenizer: PreTrainedTokenizerFast, block_size: int
) -> datasets.IterableDataset:
    """
    Load and pack a single dataset based on the info dictionary.
    """
    if "sqlite" in info["dataset_name"]:
        from datasets import Dataset
        import pandas as pd
        import sqlite3

        conn = sqlite3.connect(info["dataset_name"].replace("sqlite:///", ""))
        df = pd.read_sql_query(info["sql_query"], conn)
        conn.close()
        dataset = Dataset.from_pandas(df)
    else:
        kwargs = {"id_filter": info["id_filter"]} if "id_filter" in info else {}
        dataset = datasets.load_dataset(
            info["dataset_name"],
            info["config_name"],
            split=info["split"],
            streaming=info["streaming"] if "streaming" in info else True,
            token=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip(),
            verification_mode="no_checks",
            trust_remote_code=True,
            **kwargs,
        )
    if "should_contain" in info or "min_word_length" in info or ("blocklist_phrases" in info and "blocklist_urls" in info):
        if info.get("streaming"):
            raise Exception(
                "should_contain, min_word_length, and blocklists are only for non-streaming datasets"
            )

        def filter_func(example):
            text = example[info.get("text_column_name", "text")]
            if "should_contain" in info:
                should_contain = info["should_contain"]
                if not any(keyword in text for keyword in should_contain):
                    return False
            if "min_word_length" in info:
                min_word_length = info["min_word_length"]
                if len(text.split()) < min_word_length:
                    return False
                    
            # Apply blocklist filters if present
            if "blocklist_phrases" in info and "blocklist_urls" in info:
                # Check for blocked phrases in text
                if any(phrase.lower() in text.lower() for phrase in info["blocklist_phrases"]):
                    return False
                # Check URL in the url column if it exists
                if "url" in example:
                    if any(blocked_url in example["url"].lower() for blocked_url in info["blocklist_urls"]):
                        return False
                    
            return True

        dataset = dataset.filter(filter_func, num_proc=12)
    if info.get("shuffle_before_packing_if_not_streaming"):
        if info.get("streaming"):
            raise Exception(
                "shuffle_before_packing_if_not_streaming is only for non-streaming datasets"
            )
        print(
            f"\033[32mShuffling {info['dataset_name']} {info['config_name']} {info.get('split', '')}\033[0m"
        )
        dataset = dataset.shuffle(seed=42)
    if "max_rows" in info and info["max_rows"] is not None:
        dataset = dataset.take(info["max_rows"])
    if "custom_pack_function" in info:
        packed_dataset = info["custom_pack_function"](
            dataset, tokenizer, block_size, info.get("min_block_fraction", 0.9)
        )
        return packed_dataset

    if COLNAME not in dataset.column_names:
        dataset = dataset.rename_column(info["text_column_name"], COLNAME)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col != COLNAME]
    )

    cycled_dataset = datasets.IterableDataset.from_generator(
        generator=cycle_dataset,
        gen_kwargs={
            "dataset": dataset,
        },
    )

    packed_dataset = datasets.IterableDataset.from_generator(
        generator=pack_dataset,
        gen_kwargs={
            "dataset": cycled_dataset,
            "tokenizer": tokenizer,
            "block_size": block_size,
            "min_block_fraction": info.get("min_block_fraction", 0.0),
        },
    )

    exact_block_size = packed_dataset.filter(
        lambda x: len(x["input_ids"]) == block_size
    )
    return exact_block_size


def multi_packed_dataset_generator(
    dataset_infos: List[Dict],
    tokenizer: PreTrainedTokenizerFast,
    block_size: int = 4096,
    seed: int = 42,
) -> datasets.IterableDataset:
    """
    Generator function to first pack individual datasets and then merge them.
    This function first processes and packs each dataset specified in the `dataset_infos`
    list using the provided tokenizer and block size. Each dataset is packed such that
    sequences are marked with beginning-of-sentence (BOS) and end-of-sentence (EOS) tokens.
    It then merges these packed datasets into a single `datasets.IterableDataset` according
    to the specified weights for each dataset, ensuring a diverse and balanced stream
    of data for training or evaluation.

    Parameters:
        dataset_infos (List[Dict]): A list of dictionaries, each specifying a dataset configuration.
            Keys include 'dataset_name', 'config_name', 'split', 'text_column_name',
            'weight', 'max_rows' (optional) and 'min_block_fraction'.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use for encoding the text.
        block_size (int): The block size for packing the text data. Default is 4096.
        seed (int): A seed for random operations to ensure reproducibility. Default is 42.

    Returns:
        datasets.IterableDataset: An iterable dataset comprising merged and packed data from the specified datasets.

    Example:
        dataset_infos = [
            {
                "dataset_name": "examplecorp/newscorp_articles",
                "config_name": "simplified",
                "split": "train",
                "text_column_name": "article_body",
                "weight": 5,
                "max_rows": 100_000,
                "min_block_fraction": 0.3,
            },
            {
                "dataset_name": "globeglobe/global_news",
                "config_name": "english_only",
                "split": "validation",
                "text_column_name": "content",
                "weight": 3,
                "max_rows": 100_000,
                "min_block_fraction": 0.2,
            },
        ]

        hf_dataset_train = multi_packed_dataset_generator(
            dataset_infos=dataset_infos,
            tokenizer=tokenizer,
            block_size=1024,
            seed=42,
        )
        dataset_train = hf_dataset_train.with_format("torch")
    """
    packed_datasets = []
    probabilities = []

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    # Note that llama-3 tokenizer doesn't do this, so we need to add it manually

    total_weight = sum(info["weight"] for info in dataset_infos)

    for info in dataset_infos:
        packed_dataset = prepare_and_pack_dataset(info, tokenizer, block_size)
        if info.get("shuffle_single_packed_dataset_buffer_size"):
            shuffled_packed_dataset = packed_dataset.shuffle(
                seed=seed, buffer_size=info["shuffle_single_packed_dataset_buffer_size"]
            )
            packed_datasets.append(shuffled_packed_dataset)
        else:
            packed_datasets.append(packed_dataset)
        probability = info["weight"] / total_weight
        probabilities.append(probability)
        print(
            f"Interleaving {info['dataset_name']} {info['config_name']} {info.get('split', '')} with {probability=}"
        )

    merged_packed_dataset = datasets.interleave_datasets(
        packed_datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy="first_exhausted",
    )

    merged_packed_dataset = merged_packed_dataset.shuffle(
        seed=seed, buffer_size=5000 * len(dataset_infos)
    )
    return merged_packed_dataset
