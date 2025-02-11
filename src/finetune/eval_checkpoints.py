import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

# def format_chat_to_text(examples):
#     """Convert chat messages to formatted text using tokenizer's chat template"""
#     formatted_text = [
#         tokenizer.apply_chat_template(messages, tokenize=False)
#         for messages in examples["messages"]
#     ]
#     return {"text": formatted_text}
#
# eval_dataset_combined = eval_dataset_combined.map(
#     format_chat_to_text,
#     remove_columns=eval_dataset_combined.column_names,
#     batched=True,
# ).shuffle(seed=42)
# dataset_combined = dataset_combined.map(
#     format_chat_to_text, remove_columns=dataset_combined.column_names, batched=True
# ).shuffle(seed=42)


def fetch_checkpoint(remote_path, local_path, hostname):
    """Fetch checkpoint from remote server using scp."""
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)

        # scp the checkpoint directory
        cmd = f"scp -r {hostname}:{remote_path}/* {local_path}/"
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error fetching checkpoint: {e}")
        return False


def get_latest_checkpoint(remote_checkpoint_dir, local_checkpoint_dir, hostname):
    """Get the latest checkpoint and its step number from remote server."""
    # First, fetch the directory listing
    cmd = f"ssh {hostname} ls -1 {remote_checkpoint_dir}/checkpoint-*"
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        checkpoints = result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        return None, None

    if not checkpoints or checkpoints[0] == "":
        return None, None

    # Get the latest checkpoint by sorting based on step number
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    step = int(latest.split("-")[-1])

    # Check if checkpoint is complete by looking for done file
    done_check_cmd = f"ssh {hostname} test -f {latest}/done && echo 'exists'"
    try:
        result = subprocess.run(
            done_check_cmd, shell=True, capture_output=True, text=True
        )
        if "exists" not in result.stdout:
            return None, None
    except subprocess.CalledProcessError:
        return None, None

    # Prepare local path
    local_checkpoint_path = os.path.join(local_checkpoint_dir, f"checkpoint-{step}")

    # Fetch the checkpoint if it doesn't exist locally
    if not os.path.exists(local_checkpoint_path):
        success = fetch_checkpoint(latest, local_checkpoint_path, hostname)
        if not success:
            return None, None

    return local_checkpoint_path, step


def run_evaluation(checkpoint_path, args):
    accelerator = Accelerator()

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load evaluation datasets
    eval_dataset = (
        load_dataset(args.dataset_name_openhermes, split="train")
        .shuffle(seed=args.seed)
        .select(range(args.eval_dataset_size))
    )

    # Resume wandb run
    wandb.init(
        project="mistral-bor-1b-finetuning",
        id=args.wandb_run_id,  # Specify the run ID from training
        resume="must",
    )

    generations = []
    examples = eval_dataset.select(range(2))

    for example in examples:
        prompt_messages = []
        expected_response = None

        for msg in example["messages"]:
            if msg["role"] == "system":
                prompt_messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user" and not any(
                m["role"] == "user" for m in prompt_messages
            ):
                prompt_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant" and expected_response is None:
                expected_response = msg["content"]
                break

        if not expected_response or not prompt_messages:
            continue

        input_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(
            input_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assistant_marker = "<|assistant|>"
        last_assistant_idx = generated_text.rfind(assistant_marker)
        if last_assistant_idx != -1:
            generated_text = generated_text[
                last_assistant_idx + len(assistant_marker) :
            ].strip()

        generations.append(
            {
                "prompt": input_prompt,
                "expected": expected_response,
                "generated": generated_text,
            }
        )

    # Log to wandb with the correct step number
    step = int(checkpoint_path.split("-")[-1])
    wandb_table = wandb.Table(columns=["prompt", "expected", "generated"])
    for gen in generations:
        wandb_table.add_data(gen["prompt"], gen["expected"], gen["generated"])
    wandb.log({"eval_generations": wandb_table}, step=step)


def is_checkpoint_complete(checkpoint_path, hostname):
    """Check if checkpoint has all necessary files."""
    required_files = [
        "config.json",
        "model.safetensors",
        "adapter_config.json",  # For LoRA
        "adapter_model.safetensors",  # For LoRA
    ]

    for file in required_files:
        cmd = f"ssh {hostname} test -f {checkpoint_path}/{file} && echo 'exists'"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if "exists" not in result.stdout:
                return False
        except subprocess.CalledProcessError:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_checkpoint_dir", type=str, required=True)
    parser.add_argument("--local_checkpoint_dir", type=str, required=True)
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--wandb_run_id", type=str, required=True)
    parser.add_argument("--eval_interval", type=int, default=300)
    parser.add_argument("--cleanup_old_checkpoints", action="store_true")
    args = parser.parse_args()

    last_evaluated_checkpoint = None

    while True:
        checkpoint_path, step = get_latest_checkpoint(
            args.remote_checkpoint_dir, args.local_checkpoint_dir, args.hostname
        )

        if checkpoint_path and checkpoint_path != last_evaluated_checkpoint:
            print(f"Evaluating checkpoint: {checkpoint_path}")
            run_evaluation(checkpoint_path, args)
            last_evaluated_checkpoint = checkpoint_path

            # Optionally cleanup old checkpoints to save space
            if args.cleanup_old_checkpoints:
                for old_checkpoint in glob.glob(
                    os.path.join(args.local_checkpoint_dir, "checkpoint-*")
                ):
                    if old_checkpoint != checkpoint_path:
                        shutil.rmtree(old_checkpoint)

        time.sleep(args.eval_interval)


if __name__ == "__main__":
    main()
