import argparse
import os
from datetime import datetime


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
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the preprocessed dataset directory",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
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
        default=10,
        help="Run evaluation every X steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of steps for linear warmup",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="To use wandb or tensorboard for logging",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
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
    return parser.parse_args()


def main(args):
    import torch
    from accelerate import Accelerator
    from datasets import load_from_disk
    from tabulate import tabulate
    # from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              PreTrainedTokenizerFast, TrainerControl,
                              TrainerState, TrainingArguments)
    from transformers.trainer_callback import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    accelerator = Accelerator()

    # Add this after accelerator initialization but before loading data
    if accelerator.is_main_process:
        print("\nDistributed Setup:")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        print(f"Per device batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Total train batch size: {total_batch_size}\n")

    class GenerationEvalCallback(TrainerCallback):
        def __init__(self, eval_dataset, tokenizer, accelerator, use_wandb=False):
            self.eval_dataset = eval_dataset
            self.tokenizer = tokenizer
            self.accelerator = accelerator
            self.use_wandb = use_wandb
            if self.use_wandb:
                import wandb

        def on_evaluate(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            metrics=None,
            **kwargs,
        ):
            # Early return if not main process
            if not self.accelerator.is_main_process:
                return

            model = kwargs.get("model")
            if model is None:
                return

            # Move model to the correct device
            model = self.accelerator.unwrap_model(model)

            # Process examples in batch
            examples = self.eval_dataset.select(range(2))
            generations = []

            for example in examples:
                prompt_messages = []
                expected_response = None

                for msg in example["messages"]:
                    if msg["role"] == "system":
                        prompt_messages.append(
                            {"role": "system", "content": msg["content"]}
                        )
                    elif msg["role"] == "user" and not any(
                        m["role"] == "user" for m in prompt_messages
                    ):
                        prompt_messages.append(
                            {"role": "user", "content": msg["content"]}
                        )
                    elif msg["role"] == "assistant" and expected_response is None:
                        expected_response = msg["content"]
                        break

                if not expected_response or not prompt_messages:
                    continue

                # Create input prompt with chat template
                input_prompt = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )

                # Tokenize single prompt
                inputs = self.tokenizer(
                    input_prompt, return_tensors="pt", truncation=True, max_length=512
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

                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

                # Find the last occurrence of '<|assistant|>' and take text after it
                assistant_marker = "<|assistant|>"
                last_assistant_idx = generated_text.rfind(assistant_marker)
                if last_assistant_idx != -1:
                    generated_text = generated_text[
                        last_assistant_idx + len(assistant_marker) :
                    ].strip()

                generation = {
                    "prompt": input_prompt,
                    "expected": expected_response,
                    "generated": generated_text,
                }
                generations.append(generation)

            # Log to wandb only if enabled
            if self.use_wandb:
                wandb_table = wandb.Table(columns=["prompt", "expected", "generated"])
                for gen in generations:
                    wandb_table.add_data(
                        gen["prompt"], gen["expected"], gen["generated"]
                    )
                wandb.log({"eval_generations": wandb_table}, step=state.global_step)

            # Pretty print generations using tabulate
            table_data = [
                [i + 1, gen["prompt"], gen["expected"], gen["generated"]]
                for i, gen in enumerate(generations)
            ]
            print("\nGeneration Examples:")
            print(
                tabulate(
                    table_data,
                    headers=["#", "Prompt", "Expected", "Generated"],
                    tablefmt="grid",
                    maxcolwidths=[
                        None,
                        40,
                        40,
                        40,
                    ],  # Limit column widths for readability
                    showindex=False,
                )
            )

            if self.use_wandb:
                print(f"\nLogged {len(generations)} example generations to WandB.")

    # Initialize wandb only if specified in report_to
    if "wandb" in args.report_to and accelerator.is_main_process:
        import wandb

        wandb.init(project="mistral-bor-1b-finetuning", name="bor-1b-dutch-sft")

    # Set environment variable for experimental Flash Attention support on ROCm
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    # Load the preprocessed dataset
    dataset = load_from_disk(args.dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    if accelerator.is_main_process:
        print(f"\nDataset Overview:")
        print(f"Training examples: {len(train_dataset):,}")
        print(f"Validation examples: {len(eval_dataset):,}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_seq_length,
        padding_side="right",
        add_eos_token=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # or "<|pad|>" if you have added it to the tokenizer.

    # Load model with modified sliding window config
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=None,  # to disable bitsandbytes quantization, use accelerate for quantization
        device_map="auto",
        torch_dtype=torch.bfloat16
        if args.bf16
        else (torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        sliding_window=args.max_seq_length,
        max_position_embeddings=args.max_seq_length,
    )

    # Configure LoRA with optimized target modules
    # peft_config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=[
    #         "q_proj", "v_proj", "o_proj", "k_proj",
    #         "gate_proj", "up_proj", "down_proj",
    #     ],
    # )

    if accelerator.is_main_process:
        model_size = sum(p.numel() for p in model.parameters())
        print(f"\nModel Overview:")
        print(f"Total parameters: {model_size:,}")
        # print(f"LoRA target modules: {peft_config.target_modules}")
        # print(f"LoRA rank (r): {peft_config.r}")
        # print(f"LoRA alpha: {peft_config.lora_alpha}\n")

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
        warmup_steps=args.warmup_steps,
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
        dataloader_num_workers=min(112, os.cpu_count() or 1),
        dataloader_pin_memory=False,  # Disable pin memory when using FSDP
        dataloader_drop_last=True,
        gradient_checkpointing_kwargs=None
        if args.gradient_checkpointing
        else {"use_reentrant": False},
        remove_unused_columns=True,
        dataloader_persistent_workers=True,
        full_determinism=False,  # True enables full reproducibility but reduces performance
        torch_compile=False,  # PyTorch 2.0+ feature for potential speedup but longer startup
        packing=True,  # Efficient sequence packing for memory optimization
        dataset_batch_size=1000,  # Batch size for dataset preprocessing
        save_safetensors=True,
        logging_nan_inf_filter=True,  # Filters out NaN/Inf values in logging
        save_total_limit=2,
        save_strategy="steps",
        save_only_model=False,
        # Add these for better FSDP compatibility
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config=peft_config,
        # callbacks=[GenerationEvalCallback(eval_dataset, tokenizer, accelerator)],
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(args.output_dir)

    # Push to hub if requested
    if args.push_to_hub:
        trainer.push_to_hub()

    if "wandb" in args.report_to:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
