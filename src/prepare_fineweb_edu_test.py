from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np


def prepare_fineweb_edu():
    # Load Fineweb-edu dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", num_proc=12
    )

    # Filter for high quality content (score > 4)
    filtered = dataset.filter(lambda x: x["score"] > 4)

    # Shuffle using datasets' built-in functionality
    shuffled = filtered.shuffle(seed=42)

    # Select first 1000 examples
    test_set = shuffled.select(range(1000))

    # Push to hub
    test_set.push_to_hub(
        "fineweb_edu_score_gt_4_test", split="test", private=False, token=True
    )


if __name__ == "__main__":
    prepare_fineweb_edu()
