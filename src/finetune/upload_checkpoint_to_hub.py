from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from dotenv import load_dotenv
import torch

def upload_checkpoint_to_hub(checkpoint_dir: str, repo_id: str):
    """
    Upload checkpoint files to Hugging Face Hub using push_to_hub.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        repo_id: Hugging Face repo ID (e.g. 'username/model-name')
    """
    print(f"Loading model from {checkpoint_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    print(f"Pushing to hub {repo_id}...")
    model.push_to_hub(
        repo_id,
        safe_serialization=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer.push_to_hub(repo_id)
    
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