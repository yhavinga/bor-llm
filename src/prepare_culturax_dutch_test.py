from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np


def prepare_culturax_dutch():
    # Load Dutch subset of CulturaX
    dataset = load_dataset("uonlp/CulturaX", "nl", split="train", num_proc=12)

    # Shuffle using datasets' built-in functionality
    shuffled = dataset.shuffle(seed=42)

    # Select first 1000 examples
    test_set = shuffled.select(range(1000))

    # Push to hub
    test_set.push_to_hub(
        "culturax_dutch_test",
        private=False,
        token=True,  # Will use token from huggingface-cli login
    )


if __name__ == "__main__":
    prepare_culturax_dutch()
