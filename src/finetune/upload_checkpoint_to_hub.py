from huggingface_hub import HfApi
from pathlib import Path
import os
import argparse
from dotenv import load_dotenv

def upload_checkpoint_to_hub(checkpoint_dir: str, repo_id: str):
    """
    Upload checkpoint files to Hugging Face Hub.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        repo_id: Hugging Face repo ID (e.g. 'username/model-name') 
        token: Hugging Face API token
    """
    api = HfApi()
    
    # Create repository if it doesn't exist
    api.create_repo(repo_id=repo_id, exist_ok=False)
    
    # Files to upload (excluding optimizer states and RNG states)
    files_to_upload = [
        "config.json",
        "generation_config.json", 
        "model.safetensors",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "pytorch_model_fsdp.bin"
    ]

    checkpoint_path = Path(checkpoint_dir)
    
    # Upload each file
    for filename in files_to_upload:
        file_path = checkpoint_path / filename
        if file_path.exists():
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_id,
            )
        else:
            print(f"Warning: {filename} not found in checkpoint directory")

    print("Upload complete!")

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Upload model checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., output/model-name/checkpoint-1435)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., username/model-name)"
    )
    args = parser.parse_args()

    upload_checkpoint_to_hub(args.checkpoint_dir, args.repo_id)