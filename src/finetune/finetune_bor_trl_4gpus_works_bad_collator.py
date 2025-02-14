import os

import torch
import torch.distributed as dist
from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print("---------------------------------------------------")

# Load env and login to huggingface
load_dotenv()
# HUGGINGFACE_TOKEN = os.getenv("HF_AUTH_TOKEN")
# login(HUGGINGFACE_TOKEN)


def setup_distributed():
    """Initialize distributed training with correct GPU assignment"""
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Ensure each process gets a unique GPU
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    print(
        f"Rank {rank} using GPU {local_rank} (total GPUs: {torch.cuda.device_count()})"
    )


def prepare_data():
    dataset_path = "../dataset-bor/openhermes_leesplank_20250205_061457"
    ds = load_from_disk(dataset_path)
    print(f"Training samples: {len(ds['train'])}")
    print(f"Validation samples: {len(ds['validation'])}")
    return ds


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
        padding_side="right",
        add_eos_token=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def formatting_prompts_func(example):
        messages = example["messages"]
        if messages and isinstance(messages[0], list):
            messages = messages[0]

        chat_text = ""
        for msg in messages:
            chat_text += f"<|{msg['role']}|>\n{msg['content']}></s>"

        return [chat_text]

    # Add instruction template and modify response template handling
    instruction_template = "<|user|>\n"
    response_template = "<|assistant|>\n"
    # response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    # collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,  # Add instruction template
        response_template=response_template,
        tokenizer=tokenizer,
    )

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
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
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
