import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict
import os
from datetime import datetime

def get_project_root():
    """Get absolute path to project root from current file."""
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and save datasets for fine-tuning")
    parser.add_argument(
        "--dataset_name_openhermes",
        type=str,
        default="yhavinga/Openhermes-2.5-dutch-97k",
        help="Dataset name for OpenHermes dataset",
    )
    parser.add_argument(
        "--dataset_name_leesplank",
        type=str,
        default="UWV/Leesplank_NL_wikipedia_simplifications",
        help="Dataset name for Leesplank dataset",
    )
    parser.add_argument(
        "--use_leesplank_dataset",
        type=bool,
        default=True,
        help="Whether to use the Leesplank dataset",
    )
    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        default=256,
        help="Size of eval dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/finetune",
        help="Directory to save the processed datasets (relative to project root)",
    )
    parser.add_argument(
        "--train_dataset_size",
        type=int,
        default=50000,
        help="Size of train dataset per source (OpenHermes/Leesplank)",
    )
    return parser.parse_args()

def format_chat_leesplank(examples):
    formatted_messages = [
        [
            {"role": "user", "content": "Vereenvoudig deze tekst: " + prompt},
            {"role": "assistant", "content": result},
        ]
        for prompt, result in zip(examples["prompt"], examples["result"])
    ]
    return {"messages": formatted_messages}

def main():
    args = parse_args()
    
    # Convert output_dir to absolute path from project root and add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"openhermes{'_leesplank' if args.use_leesplank_dataset else ''}"
    output_dir = os.path.join(get_project_root(), args.output_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load and shuffle datasets
    dataset_openhermes = load_dataset(
        args.dataset_name_openhermes, split="train"
    ).shuffle(seed=args.seed)
    
    # Prepare OpenHermes datasets
    eval_dataset_openhermes = dataset_openhermes.select(range(args.eval_dataset_size))
    train_dataset_openhermes = dataset_openhermes.select(
        range(args.eval_dataset_size, args.train_dataset_size + args.eval_dataset_size)
    )

    # Remove all columns except 'messages'
    eval_dataset_openhermes = eval_dataset_openhermes.remove_columns(
        [col for col in eval_dataset_openhermes.column_names if col != "messages"]
    )
    train_dataset_openhermes = train_dataset_openhermes.remove_columns(
        [col for col in train_dataset_openhermes.column_names if col != "messages"]
    )

    if args.use_leesplank_dataset:
        # Load and process Leesplank dataset
        dataset_leesplank = load_dataset(
            args.dataset_name_leesplank, split="train"
        ).shuffle(seed=args.seed)
        
        dataset_leesplank = dataset_leesplank.select(
            range(args.train_dataset_size + args.eval_dataset_size)
        )

        # Format Leesplank dataset
        dataset_leesplank = dataset_leesplank.map(
            format_chat_leesplank,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=dataset_leesplank.column_names,
        )

        # Split Leesplank dataset
        eval_dataset_leesplank = dataset_leesplank.select(range(args.eval_dataset_size))
        train_dataset_leesplank = dataset_leesplank.select(
            range(args.eval_dataset_size, args.train_dataset_size + args.eval_dataset_size)
        )

        # Combine datasets
        eval_dataset = concatenate_datasets([eval_dataset_openhermes, eval_dataset_leesplank])
        train_dataset = concatenate_datasets([train_dataset_openhermes, train_dataset_leesplank])
    else:
        eval_dataset = eval_dataset_openhermes
        train_dataset = train_dataset_openhermes

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })

    # Save as single dataset with multiple splits
    dataset_dict.save_to_disk(output_dir)
    
    print(f"Saved dataset to {output_dir}")
    print(f"Train split size: {len(dataset_dict['train'])}")
    print(f"Validation split size: {len(dataset_dict['validation'])}")

if __name__ == "__main__":
    main() 