import os
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb


load_dotenv()  # For HF_TOKEN


def setup_distributed():
    """Initialize distributed training with correct GPU assignment"""
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Total GPUs: {torch.cuda.device_count()}")

    # Ensure each process gets a unique GPU
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    print(f"Rank {rank} using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)} (total GPUs: {torch.cuda.device_count()}) world size: {world_size}")


def prepare_data():
    dataset_path = "../dataset-bor/openhermes_leesplank_20250205_061457"
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
    
    # Initialize wandb only on main process
    if dist.get_rank() == 0:
        wandb.init(project="mistral-bor-1b-finetuning", name="bor-1b-dutch-sft-4gpus")
    
    # Load the data
    ds = prepare_data()

    output_dir = "output/bor-1b-finetune"
    
    if os.path.exists(output_dir):
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = os.path.join(output_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1])
            if dist.get_rank() == 0:
                print(f"Resuming from checkpoint: {latest_checkpoint}")
        else:
            latest_checkpoint = None
    else:
        latest_checkpoint = None

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

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=len(ds['train']) // 16 // 4,  # Total steps
        warmup_steps=20,
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        do_eval=True,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to="wandb",
        ddp_find_unused_parameters=False,
        ddp_backend='nccl',
        max_grad_norm=1.0,
        remove_unused_columns=True,
        resume_from_checkpoint=latest_checkpoint,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        ignore_data_skip=False,
        overwrite_output_dir=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        processing_class=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(ds['train'])
    metrics["val_samples"] = len(ds['validation'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if dist.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        dist.destroy_process_group()
