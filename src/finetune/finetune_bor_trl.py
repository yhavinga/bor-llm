# script_finetune.py
"""
This script fine-tunes a Mistral-based language model (yahavinga/Bor-1b)
using two datasets: yhavinga/Openhermes-2.5-dutch-46k and UWV/Leesplank_NL_wikipedia_simplifications.
It leverages QLoRA for efficient fine-tuning and supports multi-GPU training with accelerate and deepspeed.
Evaluation metrics (perplexity) and example generations are logged to Weights & Biases (WandB).
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer, SFTConfig
import wandb
from transformers.trainer_callback import TrainerCallback


# Define dataclass for training arguments
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default="yhavinga/Bor-1b",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    dataset_name_openhermes: str = field(
        default="yhavinga/Openhermes-2.5-dutch-46k",
        metadata={"help": "Dataset name for OpenHermes dataset"},
    )
    dataset_name_leesplank: str = field(
        default="UWV/Leesplank_NL_wikipedia_simplifications",
        metadata={"help": "Dataset name for Leesplank dataset"},
    )
    use_leesplank_dataset: bool = field(
        default=True, metadata={"help": "Whether to use the Leesplank dataset"}
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help": "Number of update steps to accumulate before performing a backward/update pass."
        },
    )
    learning_rate: float = field(
        default=2e-4, metadata={"help": "Initial learning rate (AdamW optimizer)."}
    )
    weight_decay: float = field(
        default=0.001, metadata={"help": "Weight decay to use."}
    )
    lora_r: int = field(default=16, metadata={"help": "Lora r dimension."})
    lora_alpha: int = field(default=32, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    output_dir: str = field(
        default="bor-1b-dutch-finetune",
        metadata={"help": "Where to store the final model."},
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the trained model to the hub."},
    )
    hub_model_id: str = field(
        default="bor-1b-dutch-finetune",
        metadata={"help": "The name of the repository to push to."},
    )
    hub_strategy: str = field(
        default="every_save", metadata={"help": "The repository to push to."}
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X updates steps."}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy to adopt during training"},
    )
    eval_steps: int = field(
        default=500, metadata={"help": "Run evaluation every X steps"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Warmup ratio for learning rate scheduler"}
    )
    report_to: str = field(
        default="wandb", metadata={"help": "To use wandb or tensorboard for logging."}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Use gradient checkpointing to save memory."}
    )
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16."})
    bf16: bool = field(default=True, metadata={"help": "Whether to use bf16."})
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. If empty string, will use ZeRO-3 and auto-config."
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization."})
    eval_dataset_size: int = field(
        default=10, metadata={"help": "Size of eval dataset for generation examples."}
    )


# Define callback for logging example generations during training
class GenerationEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, wandb_log_name="eval_generations"):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.wandb_log_name = wandb_log_name

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        generations = []
        input_texts = self.eval_dataset[
            "prompt"
        ]  # Assuming 'prompt' is the column name for prompts

        model = kwargs.get('model')
        if model is None:
            return

        for prompt in input_texts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            # Generate text
            output_ids = model.generate(
                input_ids,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
            generation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generations.append({"prompt": prompt, "generation": generation})

        # Log to wandb table
        wandb_table = wandb.Table(columns=["prompt", "generation"])
        for item in generations:
            wandb_table.add_data(item["prompt"], item["generation"])
        wandb.log({self.wandb_log_name: wandb_table})
        print(f"Logged {len(generations)} example generations to WandB.")


def main():
    script_args = ScriptArguments()

    # Initialize WandB for experiment tracking
    wandb.init(project="mistral-bor-1b-finetuning", name="bor-1b-dutch-sft")

    # Set environment variable for experimental Flash Attention support on ROCm
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.max_seq_length,
        padding_side="right",
        add_eos_token=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # or "<|pad|>" if you have added it to the tokenizer.

    # Load model with Flash Attention config
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=None,  # to disable bitsandbytes quantization, use accelerate for quantization
        device_map="auto",
        torch_dtype=torch.float16
        if script_args.fp16
        else (torch.bfloat16 if script_args.bf16 else torch.float32),
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Load and shuffle datasets with seed
    dataset_openhermes_raw = load_dataset(
        script_args.dataset_name_openhermes, split="train"
    ).shuffle(seed=script_args.seed)
    dataset_leesplank_raw = load_dataset(
        script_args.dataset_name_leesplank, split="train"
    ).shuffle(seed=script_args.seed)

    # Prepare evaluation and training datasets
    eval_dataset_openhermes = dataset_openhermes_raw.select(
        range(script_args.eval_dataset_size)
    )
    dataset_openhermes = dataset_openhermes_raw.select(
        range(script_args.eval_dataset_size, 50000 + script_args.eval_dataset_size)
    )

    # Remove all columns except 'messages' from OpenHermes dataset
    eval_dataset_openhermes = eval_dataset_openhermes.remove_columns(
        [col for col in eval_dataset_openhermes.column_names if col != "messages"]
    )
    dataset_openhermes = dataset_openhermes.remove_columns(
        [col for col in dataset_openhermes.column_names if col != "messages"]
    )

    # Format Leesplank dataset if requested (only format the rows we need)
    if script_args.use_leesplank_dataset:
        dataset_leesplank_raw = dataset_leesplank_raw.select(
            range(50000 + script_args.eval_dataset_size)
        )

        def format_chat_leesplank(examples):
            formatted_messages = [
                [
                    {"role": "user", "content": "Vereenvoudig deze tekst: " + prompt},
                    {"role": "assistant", "content": result},
                ]
                for prompt, result in zip(examples["prompt"], examples["result"])
            ]
            return {"messages": formatted_messages}

        dataset_leesplank = dataset_leesplank_raw.map(
            format_chat_leesplank,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=dataset_leesplank_raw.column_names,
        )

        eval_dataset_leesplank = dataset_leesplank.select(
            range(script_args.eval_dataset_size)
        )
        dataset_leesplank = dataset_leesplank.select(
            range(script_args.eval_dataset_size, 50000 + script_args.eval_dataset_size)
        )
        eval_dataset_generation = concatenate_datasets(
            [eval_dataset_openhermes, eval_dataset_leesplank]
        )
        dataset_combined = concatenate_datasets([dataset_openhermes, dataset_leesplank])
    else:
        eval_dataset_generation = eval_dataset_openhermes
        dataset_combined = dataset_openhermes

    def format_chat_to_text(examples):
        """Convert chat messages to formatted text using tokenizer's chat template"""
        formatted_text = [
            tokenizer.apply_chat_template(messages, tokenize=False)
            for messages in examples["messages"]
        ]
        return {"text": formatted_text}

    eval_dataset_generation = eval_dataset_generation.map(
        format_chat_to_text,
        remove_columns=eval_dataset_generation.column_names,
        batched=True,
    ).shuffle(seed=42)
    dataset_combined = dataset_combined.map(
        format_chat_to_text, remove_columns=dataset_combined.column_names, batched=True
    ).shuffle(seed=42)

    # First create SFTConfig with all training arguments
    training_args = SFTConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        num_train_epochs=script_args.num_train_epochs,
        warmup_ratio=script_args.warmup_ratio,
        report_to=script_args.report_to,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        hub_strategy=script_args.hub_strategy,
        gradient_checkpointing=script_args.gradient_checkpointing,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        deepspeed=script_args.deepspeed,
        seed=script_args.seed,
        max_seq_length=script_args.max_seq_length,
        # Additional optimizations
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=0,
        dataloader_num_workers=os.cpu_count(),
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=True,
        dataloader_persistent_workers=True,
        full_determinism=False,
        torch_compile=False,  # Set to True if using PyTorch 2.0+ and want to use compiled mode
        packing=True,  # Since we're using SFTTrainer
        dataset_batch_size=1000,
        save_safetensors=True,
        logging_nan_inf_filter=True,
        save_total_limit=2,  # Keep only the last 2 checkpoints
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset_combined,
        eval_dataset=eval_dataset_generation,
        peft_config=peft_config,
        callbacks=[GenerationEvalCallback(eval_dataset_generation, tokenizer)],
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(script_args.output_dir)

    # Push to hub if requested
    if script_args.push_to_hub:
        trainer.push_to_hub()

    wandb.finish()


if __name__ == "__main__":
    main()
