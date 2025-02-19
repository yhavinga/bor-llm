import os

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["ROCBLAS_USE_HIPBLASLT"] = "1"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "0"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["TORCH_NCCL_HIGH_PRIORITY"] = "0"

import torch
import torch.distributed as dist
import wandb
from datasets import load_from_disk, load_dataset, DatasetDict
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import datetime

# Constants
MAX_SEQUENCE_LENGTH = 4096
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 1
WARMUP_STEPS = 100
LOGGING_STEPS = 10
EVAL_STEPS = 500
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 3
MAX_GRAD_NORM = 1.0
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
            rank=rank,
            timeout=datetime.timedelta(minutes=30)
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
    # dataset_path = "./dataset/finetune/openhermes_leesplank_20250218_022850"
    # dataset = load_from_disk(dataset_path)
    dataset = load_dataset("yhavinga/Leesplank_NL_wikipedia_simplifications_preprocessed_chatml_format")
    
    # Keep only the 'messages' column
    dataset = dataset.remove_columns([col for col in dataset['train'].column_names if col != 'messages'])

    # Create validation set from original training data
    original_train = dataset['train']
    validation_dataset = original_train.select(range(1024))  # First 1024 for validation
    
    # Create new training set from remaining data (preserves original order)
    train_dataset = original_train.select(range(1024, len(original_train)))
    
    # Build new DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })

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
    output_dir = "output/bor-1b-finetune2"

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
    # print the device of the model
    print(f"Model is on device: {model.device}")

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
        do_eval=True,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        max_grad_norm=MAX_GRAD_NORM,
        remove_unused_columns=True,
        resume_from_checkpoint=latest_checkpoint,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        ignore_data_skip=False,
        overwrite_output_dir=False,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)

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
