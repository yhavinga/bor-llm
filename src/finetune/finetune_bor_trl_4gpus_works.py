import os

import torch
import torch.distributed as dist
from datasets import load_from_disk
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print("---------------------------------------------------")

load_dotenv()  # For HF_TOKEN


def setup_distributed():
    """Initialize distributed training with correct GPU assignment"""
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Ensure each process gets a unique GPU
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    print(
        f"Rank {rank} using GPU {local_rank} (total GPUs: {torch.cuda.device_count()}) world size: {world_size}"
    )


def prepare_data():
    # dataset_path = "../dataset-bor/openhermes_leesplank_20250205_061457"
    dataset_path = "./dataset/finetune/openhermes_leesplank_20250209_074026"
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    if dist.get_rank() == 0:
        print(f"\nDataset Overview:")
        print(f"Training examples: {len(train_dataset):,}")
        print(f"Validation examples: {len(eval_dataset):,}\n")

    return {"train": train_dataset, "validation": eval_dataset}


def main():
    setup_distributed()

    # Check if GPU assignment is correct
    print(f"Process {dist.get_rank()} is using GPU {torch.cuda.current_device()}")

    # Load the data
    ds = prepare_data()

    # Get tokenizer with proper configuration
    model_id = "yhavinga/Bor-1b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=4096,
        padding_side="left",
        add_eos_token=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        # sliding_window=2048,
        # max_position_embeddings=2048,
    )

    output_dir = "qlora_output/bor-1b-finetune"
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=len(ds["train"]) // 16 // 4,
        warmup_steps=20,
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=100,
        do_eval=True,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        max_grad_norm=1.0,
        # Remove unused columns to let TRL handle the chat template
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    metrics = train_result.metrics
    metrics["train_samples"] = len(ds["train"])
    metrics["val_samples"] = len(ds["validation"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    try:
        main()
    finally:
        dist.destroy_process_group()
