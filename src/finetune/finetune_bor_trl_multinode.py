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
LEARNING_RATE = 1e-5
PER_DEVICE_BATCH_SIZE = 16
NUM_EPOCHS = 3
WARMUP_STEPS = 100
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 10000
SAVE_TOTAL_LIMIT = 3
MAX_GRAD_NORM = 2.0
MAX_STEPS = 50000

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


def find_latest_valid_checkpoint(output_dir):
    """Find the latest checkpoint that has a valid trainer_state.json file"""
    if not os.path.exists(output_dir):
        return None

    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
    if not checkpoints:
        return None

    # Sort checkpoints by number and check for trainer_state.json
    valid_checkpoints = []
    for checkpoint in sorted(checkpoints, key=lambda x: int(x.split("-")[1])):
        checkpoint_path = os.path.join(output_dir, checkpoint)
        if os.path.exists(os.path.join(checkpoint_path, "trainer_state.json")):
            valid_checkpoints.append(checkpoint_path)

    return valid_checkpoints[-1] if valid_checkpoints else None


def main():
    setup_distributed()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    local_world_size = torch.cuda.device_count()  # GPUs per node
    node_rank = rank // local_world_size  # Which node we're on

    # Set device before any tensor operations
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True  # Optimize kernel selection

    # Initialize wandb only on rank 0
    is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
    if rank == 0:
        wandb.init(project="mistral-bor-1b-finetuning")
        if is_sweep:
            config = wandb.config
            learning_rate = config.learning_rate
            batch_size = config.batch_size
            max_grad_norm = config.max_gradient_norm
            warmup_steps = config.warmup_steps
            max_steps = config.max_steps
        else:
            learning_rate = LEARNING_RATE
            batch_size = PER_DEVICE_BATCH_SIZE
            max_grad_norm = MAX_GRAD_NORM
            warmup_steps = WARMUP_STEPS
            max_steps = MAX_STEPS
            wandb.config.update({
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_gradient_norm": max_grad_norm,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps
            })
    else:
        # Initialize with default values for non-primary processes
        learning_rate = LEARNING_RATE
        batch_size = PER_DEVICE_BATCH_SIZE
        max_grad_norm = MAX_GRAD_NORM
        warmup_steps = WARMUP_STEPS
        max_steps = MAX_STEPS

    # Broadcast hyperparameters from rank 0 to all processes
    learning_rate = torch.tensor(learning_rate).cuda()
    batch_size = torch.tensor(batch_size).cuda()
    max_grad_norm = torch.tensor(max_grad_norm).cuda()
    warmup_steps = torch.tensor(warmup_steps).cuda()
    max_steps = torch.tensor(max_steps).cuda()
    
    dist.broadcast(learning_rate, 0)
    dist.broadcast(batch_size, 0)
    dist.broadcast(max_grad_norm, 0)
    dist.broadcast(warmup_steps, 0)
    dist.broadcast(max_steps, 0)

    # Convert back to Python scalars
    learning_rate = learning_rate.item()
    batch_size = int(batch_size.item())
    max_grad_norm = max_grad_norm.item()
    warmup_steps = int(warmup_steps.item())
    max_steps = int(max_steps.item())

    ds = prepare_data()
    output_dir = "output/bor-1b-finetune"

    # Only load checkpoint if not in sweep
    latest_checkpoint = None if is_sweep else find_latest_valid_checkpoint(output_dir)
    if latest_checkpoint and dist.get_rank() == 0:
        print(f"Resuming from checkpoint: {latest_checkpoint}")

    model_id = "yhavinga/Bor-1b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="left",  # left padding for Mistral like tokenizer
        add_eos_token=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # First load model to CPU, then move to correct GPU
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

    # Define FSDP config with hybrid sharding strategy
    fsdp_config = {
        "sharding_strategy": ShardingStrategy.HYBRID_SHARD,  # Shard within node, replicate across nodes
        
        # Optimize communication
        "backward_prefetch": "backward_post",  # More aggressive prefetch
        "forward_prefetch": True,  # Enable forward prefetch for better overlap
        "limit_all_gathers": True,
        
        # Wrap larger chunks to reduce communication frequency
        "transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
        "min_num_params": int(5e7),  # Larger minimum size for wrapping
        
        # Memory optimizations
        "activation_checkpointing": True,
        "activation_checkpoint_interval": 2,  # Checkpoint every 2 layers instead of every layer
        "use_orig_params": False,

        # Only rank 0 loads model initially, others get weights via sync
        # Significantly reduces CPU memory usage during loading
        "cpu_ram_efficient_loading": True,

        # Prevents too many in-flight all-gathers by adding synchronization
        # Reduces memory spikes at cost of slight performance impact
        "limit_all_gathers": True,
    }

    total_train_steps = min(
        MAX_STEPS, NUM_EPOCHS * len(ds["train"]) // (PER_DEVICE_BATCH_SIZE * world_size)
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps" if not is_sweep else "no",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT if not is_sweep else None,

        # Updated FSDP settings
        fsdp=["hybrid_shard", "auto_wrap"],  # Use hybrid sharding
        fsdp_config=fsdp_config,
        
        # Precision and Performance
        do_eval=True,
        lr_scheduler_type="constant_with_warmup" if is_sweep else "cosine",
        fp16=False,  # Don't use FP16 with BF16
        bf16=True,   # Use BF16 on AMD MI300X
        
        # configured with fsdp_config
        # gradient_checkpointing=True,  # Enable gradient checkpointing
        # gradient_checkpointing_kwargs={"use_reentrant": False},  # Use non-reentrant for better performance
        
        # Distributed Training
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        max_grad_norm=max_grad_norm,
        
        # Checkpoint and Saving
        remove_unused_columns=True,
        resume_from_checkpoint=latest_checkpoint,
        save_safetensors=True,
        load_best_model_at_end=False if is_sweep else True,
        metric_for_best_model="eval_loss",
        ignore_data_skip=False,
        overwrite_output_dir=False,
        
        # Only report metrics from rank 0
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

    # Save model and metrics only on rank 0 and if not in sweep
    if dist.get_rank() == 0 and not is_sweep:
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
