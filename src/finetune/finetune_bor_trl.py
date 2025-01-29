import argparse
from datetime import datetime
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Mistral-based language model using QLoRA"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="yhavinga/Bor-1b",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name_openhermes",
        type=str,
        default="yhavinga/Openhermes-2.5-dutch-97k",
        help="Dataset name for OpenHermes dataset",
    )
    parser.add_argument(
        "--dataset_name_leesplank",
        type=str,
        default="UWV/Leesplank_NL_wikipedia_simplifications",
        help="Dataset name for Leesplank dataset",
    )
    parser.add_argument(
        "--use_leesplank_dataset",
        type=bool,
        default=True,
        help="Whether to use the Leesplank dataset",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of update steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (AdamW optimizer)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay to use",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA attention dimension. Lower values = smaller size but less capacity, higher values = larger size but more capacity. Common values: 8, 16, 32, 64",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA scaling factor. Usually set to 2x lora_r. Higher values = stronger updates. Common values: 16, 32, 64",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers. Helps prevent overfitting. Common values: 0.05-0.1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bor-1b-dutch-finetune",
        help="Where to store the final model",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the trained model to the hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="bor-1b-dutch-finetune",
        help="The name of the repository to push to",
    )
    parser.add_argument(
        "--hub_strategy",
        type=str,
        default="every_save",
        help="The repository to push to",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy to adopt during training",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X steps",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="To use wandb or tensorboard for logging",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use fp16",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Whether to use bf16",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to deepspeed config if using deepspeed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        default=48,
        help="Size of eval dataset for generation examples",
    )
    return parser.parse_args()


def main(args):
    import torch
    import wandb
    from datasets import concatenate_datasets, load_dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              PreTrainedTokenizerFast, TrainerControl,
                              TrainerState, TrainingArguments)
    from transformers.trainer_callback import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from accelerate import Accelerator
    from tabulate import tabulate

    accelerator = Accelerator()

    class GenerationEvalCallback(TrainerCallback):
        def __init__(self, eval_dataset, tokenizer, accelerator):
            self.eval_dataset = eval_dataset
            self.tokenizer = tokenizer
            self.accelerator = accelerator

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
            if not self.accelerator.is_main_process:
                return

            model = kwargs.get('model')
            if model is None:
                return

            # Process examples in batch
            examples = self.eval_dataset.select(range(2))
            generations = []
            
            for example in examples:
                prompt_messages = []
                expected_response = None
                
                for msg in example['messages']:
                    if msg["role"] == "system":
                        prompt_messages.append({"role": "system", "content": msg["content"]})
                    elif msg["role"] == "user" and not any(m["role"] == "user" for m in prompt_messages):
                        prompt_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant" and expected_response is None:
                        expected_response = msg["content"]
                        break

                if not expected_response or not prompt_messages:
                    continue

                # Create input prompt with chat template
                input_prompt = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Tokenize single prompt
                inputs = self.tokenizer(
                    input_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(model.device)

                # Generate for single prompt
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50,
                )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Find the last occurrence of '<|assistant|>' and take text after it
                assistant_marker = '<|assistant|>'
                last_assistant_idx = generated_text.rfind(assistant_marker)
                if last_assistant_idx != -1:
                    generated_text = generated_text[last_assistant_idx + len(assistant_marker):].strip()
                
                generation = {
                    "prompt": input_prompt,
                    "expected": expected_response,
                    "generated": generated_text
                }
                generations.append(generation)

            # Log to wandb
            wandb_table = wandb.Table(columns=["prompt", "expected", "generated"])
            for gen in generations:
                wandb_table.add_data(gen["prompt"], gen["expected"], gen["generated"])
            wandb.log({"eval_generations": wandb_table}, step=state.global_step)
            
            # Pretty print generations using tabulate
            table_data = [[i+1, gen["prompt"], gen["expected"], gen["generated"]] 
                         for i, gen in enumerate(generations)]
            print("\nGeneration Examples:")
            print(tabulate(
                table_data,
                headers=["#", "Prompt", "Expected", "Generated"],
                tablefmt="grid",
                maxcolwidths=[None, 40, 40, 40],  # Limit column widths for readability
                showindex=False
            ))
            
            print(f"\nLogged {len(generations)} example generations to WandB.")

    if accelerator.is_main_process:
        wandb.init(project="mistral-bor-1b-finetuning", name="bor-1b-dutch-sft")

    # Set environment variable for experimental Flash Attention support on ROCm
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_seq_length,
        padding_side="left",
        add_eos_token=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # or "<|pad|>" if you have added it to the tokenizer.

    # Load model with Flash Attention config
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=None,  # to disable bitsandbytes quantization, use accelerate for quantization
        device_map="cuda",
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Configure LoRA with optimized target modules
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "v_proj", "o_proj", "k_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    if accelerator.is_main_process:
        model_size = sum(p.numel() for p in model.parameters())
        print(f"\nModel Overview:")
        print(f"Total parameters: {model_size:,}")
        print(f"LoRA target modules: {peft_config.target_modules}")
        print(f"LoRA rank (r): {peft_config.r}")
        print(f"LoRA alpha: {peft_config.lora_alpha}\n")

    # Load and shuffle datasets with seed
    dataset_openhermes_raw = load_dataset(
        args.dataset_name_openhermes, split="train"
    ).shuffle(seed=args.seed)
    dataset_leesplank_raw = load_dataset(
        args.dataset_name_leesplank, split="train"
    ).shuffle(seed=args.seed)

    # Prepare evaluation and training datasets
    eval_dataset_openhermes = dataset_openhermes_raw.select(
        range(args.eval_dataset_size)
    )
    dataset_openhermes = dataset_openhermes_raw.select(
        range(args.eval_dataset_size, 50000 + args.eval_dataset_size)
    )

    # Remove all columns except 'messages' from OpenHermes dataset
    eval_dataset_openhermes = eval_dataset_openhermes.remove_columns(
        [col for col in eval_dataset_openhermes.column_names if col != "messages"]
    )
    dataset_openhermes = dataset_openhermes.remove_columns(
        [col for col in dataset_openhermes.column_names if col != "messages"]
    )

    # Format Leesplank dataset if requested (only format the rows we need)
    if args.use_leesplank_dataset:
        dataset_leesplank_raw = dataset_leesplank_raw.select(
            range(50000 + args.eval_dataset_size)
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
            range(args.eval_dataset_size)
        )
        dataset_leesplank = dataset_leesplank.select(
            range(args.eval_dataset_size, 50000 + args.eval_dataset_size)
        )
        eval_dataset_combined = concatenate_datasets(
            [eval_dataset_openhermes, eval_dataset_leesplank]
        )
        dataset_combined = concatenate_datasets([dataset_openhermes, dataset_leesplank])
    else:
        eval_dataset_combined = eval_dataset_openhermes
        dataset_combined = dataset_openhermes

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

    # First create SFTConfig with all training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        run_name=f"bor-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=0,  # Number of steps for learning rate warmup. Alternative to warmup_ratio
        dataloader_num_workers=os.cpu_count(),
        dataloader_pin_memory=True,  # Pins memory for faster data transfer to GPU
        dataloader_drop_last=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Memory optimization setting
        remove_unused_columns=True,
        dataloader_persistent_workers=True,
        full_determinism=False,  # True enables full reproducibility but reduces performance
        torch_compile=False,  # PyTorch 2.0+ feature for potential speedup but longer startup
        packing=True,  # Efficient sequence packing for memory optimization
        dataset_batch_size=1000,  # Batch size for dataset preprocessing
        save_safetensors=True,
        logging_nan_inf_filter=True,  # Filters out NaN/Inf values in logging
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset_combined,
        eval_dataset=eval_dataset_combined,
        peft_config=peft_config,
        callbacks=[GenerationEvalCallback(eval_dataset_combined, tokenizer, accelerator)],
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(args.output_dir)

    # Push to hub if requested
    if args.push_to_hub:
        trainer.push_to_hub()

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
