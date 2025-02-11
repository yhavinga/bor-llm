import os

import torch
import torch.distributed as dist
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)

# Constants
MAX_SEQUENCE_LENGTH = 4096
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
WARMUP_STEPS = 100
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
MAX_GRAD_NORM = 1.0
MAX_STEPS = 4000

load_dotenv()  # For HF_TOKEN


def setup_distributed():
    """Initialize distributed training with correct GPU assignment"""
    try:
        # Get environment variables for distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        # Set the device before initializing process group
        torch.cuda.set_device(local_rank)

        # Initialize process group with explicit parameters
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )

        if rank == 0:
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"Total GPUs: {torch.cuda.device_count()}")
            print(f"World size: {world_size}")

        print(
            f"Rank {rank} (Local Rank {local_rank}) using GPU: {torch.cuda.get_device_name(local_rank)}"
        )

    except ValueError as e:
        if "RANK" in str(e):
            script_path = os.path.abspath(__file__) if '__file__' in globals() else "script"
            error_msg = (
                f"{e}\n\n"
                "For single-node multi-GPU:\n"
                f"\033[5;33mtorchrun --nproc_per_node=4 {script_path}\033[0m\n\n"
                "For multi-node (example for 2 nodes with 4 GPUs each):\n"
                f"\033[5;33mtorchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=job_1 --rdzv_backend=c10d --rdzv_endpoint=MASTER_NODE_IP:PORT {script_path}\033[0m\n"
            )
            raise ValueError(error_msg)
        else:
            raise


def prepare_data():
    # dataset_path = "../dataset-bor/openhermes_leesplank_20250205_061457"
    dataset_path = "./dataset/finetune/openhermes_leesplank_20250209_074026"
    dataset = load_from_disk(dataset_path)

    if dist.get_rank() == 0:
        print(
            f"\nDataset Overview:\nTraining examples: {len(dataset['train']):,}\nValidation examples: {len(dataset['validation']):,}\n"
        )

    return dataset


def main():
    setup_distributed()
    world_size = dist.get_world_size()

    # Initialize wandb only on main process
    if dist.get_rank() == 0:
        wandb.init(project="mistral-bor-1b-finetuning", name="bor-1b-dutch-sft-4gpus")

    ds = prepare_data()
    output_dir = "output/bor-1b-finetune"

    if os.path.exists(output_dir):
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = os.path.join(
                output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            )
            if dist.get_rank() == 0:
                print(f"Resuming from checkpoint: {latest_checkpoint}")
        else:
            latest_checkpoint = None
    else:
        latest_checkpoint = None

    model_id = "yhavinga/Bor-1b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="left",  # left padding for Mistral like tokenizer
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
        use_cache=False
        # sliding_window=2048,
        # max_position_embeddings=2048,
    )

    # Enable gradient checkpointing before FSDP wrapping
    model.gradient_checkpointing_enable()

    # Define FSDP config
    fsdp_config = {
        # Specify which layer class to wrap with FSDP
        # Alternative: Use min_num_params instead for automatic wrapping based on parameter count
        "transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],

        # Controls when to prefetch next parameters during backward pass
        # "backward_pre": Prefetch before current gradient computation (better memory, default)
        # "backward_post": Prefetch after current gradient computation (faster but more memory)
        "backward_prefetch": "backward_pre",

        # Whether to prefetch parameters during forward pass
        # False saves memory, True may improve speed but uses more memory
        "forward_prefetch": False,

        # Enables activation checkpointing to trade compute for memory savings
        # Critical for training large models with limited memory
        "activation_checkpointing": True,

        # Ensures all GPUs start with identical weights by broadcasting from rank 0
        # Must be True when using cpu_ram_efficient_loading
        "sync_module_states": True,

        # If True, allows mixing frozen and trainable parameters
        # Set False to save memory when all parameters are trainable
        "use_orig_params": False,

        # Only rank 0 loads model initially, others get weights via sync
        # Significantly reduces CPU memory usage during loading
        "cpu_ram_efficient_loading": True,

        # Prevents too many in-flight all-gathers by adding synchronization
        # Reduces memory spikes at cost of slight performance impact
        "limit_all_gathers": True,
    }

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=NUM_EPOCHS,
        max_steps=min(
            MAX_STEPS, len(ds["train"]) // BATCH_SIZE // world_size
        ),  # Total steps
        warmup_steps=WARMUP_STEPS,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        
        # FSDP Configuration - new style
        fsdp=["full_shard", "auto_wrap"],  # Enable FSDP with auto wrapping
        fsdp_config=fsdp_config,  # Use the new config dict
        
        # Precision and Performance
        do_eval=True,
        lr_scheduler_type="cosine",
        fp16=False,  # Don't use FP16 with BF16
        bf16=True,   # Use BF16 on AMD MI300X
        
        # configured with fsdp_config
        # gradient_checkpointing=True,  # Enable gradient checkpointing
        # gradient_checkpointing_kwargs={"use_reentrant": False},  # Use non-reentrant for better performance
        
        # Distributed Training
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        max_grad_norm=MAX_GRAD_NORM,
        
        # Checkpoint and Saving
        remove_unused_columns=True,
        resume_from_checkpoint=latest_checkpoint,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        ignore_data_skip=False,
        overwrite_output_dir=False,
        
        # Report to wandb from main process only
        report_to="wandb" if dist.get_rank() == 0 else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )

    # Optional: Configure optimizer for BF16
    # This ensures optimizer states are in BF16 to save memory
    # Note: This is handled automatically by Trainer when bf16=True
    """
    optimizer = trainer.optimizer
    for group in optimizer.state.values():
        for state_key, state_value in group.items():
            if isinstance(state_value, torch.Tensor):
                state_value.data = state_value.data.to(torch.bfloat16)
    """

    train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)

    # Save model and metrics only on rank 0
    if dist.get_rank() == 0:
        metrics = train_result.metrics
        metrics["train_samples"] = len(ds["train"])
        metrics["val_samples"] = len(ds["validation"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Ensure we save the full (unsharded) model
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        dist.destroy_process_group()
