from datasets import load_dataset, Dataset
from huggingface_hub import login
import json
import random


def convert_conversation(conversations):
    messages = []
    for msg in conversations:
        role = "user" if msg["from"] == "human" else "assistant"
        messages.append({"role": role, "content": msg["value"]})
    return {"messages": messages}


def process_dataset():
    # Load original dataset
    dataset = load_dataset("yhavinga/Openhermes-2.5-dutch-46k")

    # Convert conversations
    converted_data = []
    for item in dataset["train"]:
        try:
            conversations = json.loads(item["conversations_nl"])
            converted = convert_conversation(conversations)
            converted_data.append(converted)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping malformed entry: {e}")
            continue

    # Randomly select 1000 examples for test split
    random.seed(42)  # For reproducibility
    test_indices = set(random.sample(range(len(converted_data)), 1000))

    train_data = [
        item for i, item in enumerate(converted_data) if i not in test_indices
    ]
    test_data = [item for i, item in enumerate(converted_data) if i in test_indices]

    # Create new datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # Create DatasetDict
    from datasets import DatasetDict

    combined_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Push to hub
    login()
    combined_dataset.push_to_hub(
        "yhavinga/openhermes-dutch-sft",
        private=False,
        commit_message="Add test split with 1000 random examples",
    )


if __name__ == "__main__":
    process_dataset()
