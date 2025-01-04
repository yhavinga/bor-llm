import datasets
from transformers import AutoTokenizer
import random
from colorama import init, Fore, Back, Style
import os
from pathlib import Path

# Initialize colorama for Windows compatibility
init()

# Find the most recent directory in last_run_prepared
last_run_dir = Path("last_run_prepared")
if not last_run_dir.exists():
    raise FileNotFoundError("last_run_prepared directory not found")

latest_dir = max(
    (d for d in last_run_dir.iterdir() if d.is_dir()),
    key=lambda x: x.stat().st_mtime
)

# Load the dataset from the latest directory
dataset = datasets.load_from_disk(str(latest_dir))
tokenizer = AutoTokenizer.from_pretrained("yhavinga/bor-420k-32k-110k-158k-108k-250k")

random_idx = random.randint(0, len(dataset) - 1)
sample = dataset[random_idx]

# Decode and align tokens with labels
tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
labels = sample['labels']
attention_mask = sample['attention_mask']

print("\nFull text with color coding:")
print("-" * 100)
print(f"Total tokens: {len(tokens)}")
print(f"Active tokens (non-masked): {sum(attention_mask)}")
print(f"{Fore.BLUE}Blue: Input text (not trained on)")
print(f"{Fore.GREEN}Green: Target text (trained on)")
print(f"{Fore.LIGHTBLACK_EX}Gray: Masked/padding tokens{Style.RESET_ALL}")
print("-" * 100)

# Print full text with colors
for i, (token, label, mask) in enumerate(zip(tokens, labels, attention_mask)):
    if mask == 0:
        # Masked/padding token
        print(Fore.LIGHTBLACK_EX + token + Style.RESET_ALL, end='')
    elif label == -100:
        # Input text
        print(Fore.BLUE + token + Style.RESET_ALL, end='')
    else:
        # Target text
        print(Fore.GREEN + token + Style.RESET_ALL, end='')
print("\n")