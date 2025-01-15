import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch.cuda
import json
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_word_level_perplexity_v2(
    model, tokenizer, context_length, text, device="cuda"
):
    """
    Computes word-level metrics including:
    1. Token-level perplexity
    2. Cumulative Word-Level Bits (CWB)

    Token-level perplexity = exp(-1/N ∑_{i=1}^N log_e P(token_i))
    where N is total number of tokens and P(token) is the model's probability for each token

    Cumulative Word Bits (CWB):
    CWB = (1/W) * -log₂(∏_{i=1}^N P(token_i))
        = (1/W) * ∑_{i=1}^N -log_e P(token_i) / log_e(2)
    Where W is the total number of words in the text.
    """
    encoded = tokenizer(
        text,
        max_length=context_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    num_tokens = attention_mask.sum().item()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits.to(device)

    # Calculate token-level metrics using log_softmax
    next_token_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:].to(device)
    shifted_attention_mask = attention_mask[:, 1:].to(device)

    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    seq_token_log_probs = log_probs[
        0, torch.arange(shifted_input_ids.size(1)).to(device), shifted_input_ids[0]
    ]

    # Calculate token-level perplexity
    masked_log_probs = seq_token_log_probs * shifted_attention_mask[0]
    total_log_prob = masked_log_probs.sum()
    token_perplexity = torch.exp(-total_log_prob / shifted_attention_mask.sum()).item()

    # Convert natural log sum to base-2 bits
    total_shannen_bits = -total_log_prob / torch.log(torch.tensor(2.0))
    num_words = len(tokenizer.decode(input_ids[0]).split())
    mean_shannen_bits = (total_shannen_bits / num_words).item()

    return token_perplexity, num_tokens, mean_shannen_bits, num_words


def calculate_perplexities(model, tokenizer, dataset, context_length):
    token_ppls = []
    total_tokens = 0
    cumulative_word_bits_list = []
    detailed_metrics = []

    with tqdm(dataset, desc="Processing texts") as pbar:
        for i, row in enumerate(pbar):
            (
                token_ppl,
                num_tokens,
                mean_bits,
                num_words,
            ) = calculate_word_level_perplexity_v2(
                model, tokenizer, context_length, row["text"], device=DEVICE
            )

            detailed_metrics.append(
                {
                    "text_id": i,
                    "token_perplexity": token_ppl,
                    "cumulative_word_bits": mean_bits,
                    "num_tokens": num_tokens,
                    "num_words": num_words,
                }
            )

            token_ppls.append(token_ppl)
            cumulative_word_bits_list.append(mean_bits)
            total_tokens += num_tokens

            pbar.set_postfix(
                {
                    "tokens": total_tokens,
                    "token_ppl": f"{token_ppl:.1f}",
                    "bits": f"{mean_bits:.1f}",
                }
            )

    valid_token_ppls = [x for x in token_ppls if not (np.isnan(x) or np.isinf(x))]

    def tensor_to_float(x):
        """Helper function to safely convert tensor/scalar to float."""
        if torch.is_tensor(x):
            return float(x.float().cpu().numpy())  # Convert bfloat16 to float32 first
        return float(x)

    if valid_token_ppls:
        token_percentiles = np.percentile(
            [tensor_to_float(x) for x in valid_token_ppls], [25, 50, 75]
        )
        mean_token_perplexity = np.mean([tensor_to_float(x) for x in valid_token_ppls])
        mean_cumulative_word_bits = np.mean(
            [tensor_to_float(x) for x in cumulative_word_bits_list]
        )
    else:
        token_percentiles = [float("nan")] * 3
        mean_token_perplexity = float("nan")
        mean_cumulative_word_bits = float("nan")

    return {
        "aggregated_metrics": {
            "token_perplexity": {
                "25": float(token_percentiles[0]),
                "50": float(token_percentiles[1]),
                "75": float(token_percentiles[2]),
            },
            "mean_token_perplexity": float(mean_token_perplexity),
            "mean_cumulative_word_bits": float(mean_cumulative_word_bits),
        },
        "text_level_metrics": [
            {
                "text_id": m["text_id"],
                "token_perplexity": tensor_to_float(m["token_perplexity"]),
                "cumulative_word_bits": tensor_to_float(m["cumulative_word_bits"]),
                "num_tokens": m["num_tokens"],
                "num_words": m["num_words"],
            }
            for m in detailed_metrics
        ],
    }


def main():
    print("Loading datasets...")
    debug = False
    # debug = True

    if debug:
        datasets = {
            "fineweb_edu_4": load_dataset(
                "yhavinga/fineweb_edu_score_gt_4_test", split="test", num_proc=16
            ).select(range(40)),
            # "culturax_nl": load_dataset(
            #     "yhavinga/culturax_dutch_test", split="train", num_proc=16
            # ).select(range(40))
        }
        models = {
            "gpt-neo-125M-dutch": {
                "path": "yhavinga/gpt-neo-125M-dutch",
                "context_length": 512,
            },
        }
    else:
        datasets = {
            "fineweb_edu_4": load_dataset(
                "yhavinga/fineweb_edu_score_gt_4_test", split="test", num_proc=16
            ),
            "mc4_nl": load_dataset(
                "yhavinga/mc4_nl_cleaned",
                "tiny",
                split="validation[:1000]",
                num_proc=16,
            ),
            "culturax_nl": load_dataset(
                "yhavinga/culturax_dutch_test", split="train", num_proc=16
            ),
        }

        models = {
            "gpt-neo-125M-dutch": {
                "path": "yhavinga/gpt-neo-125M-dutch",
                "context_length": 512,
            },
            "gpt2-medium-dutch": {
                "path": "yhavinga/gpt2-medium-dutch",
                "context_length": 512,
            },
            "gpt2-large-dutch": {
                "path": "yhavinga/gpt2-large-dutch",
                "context_length": 512,
            },
            "Bor-1B": {"path": "yhavinga/Bor-1B", "context_length": 4096},
            "gpt-neo-1.3B-dutch": {
                "path": "yhavinga/gpt-neo-1.3B-dutch",
                "context_length": 512,
            },
            "Llama-3.2-1B": {"path": "meta-llama/Llama-3.2-1B", "context_length": 4096},
            "Phi-3.5-mini-instruct": {
                "path": "microsoft/Phi-3.5-mini-instruct",
                "context_length": 4096,
            },
            "Fietje-2": {"path": "BramVanroy/fietje-2", "context_length": 2048},
            "SmolLM2-1.7B": {
                "path": "HuggingFaceTB/SmolLM2-1.7B",
                "context_length": 4096,
            },
        }

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_flash_attention_2": False,
    }

    for model_name, model_info in models.items():
        # Enable flash attention for non-GPT models
        model_kwargs["use_flash_attention_2"] = "gpt" not in model_name.lower()

        safe_model_name = model_name.replace("/", "_")
        output_file = (
            f'{safe_model_name}_evaluation_results{"_debug" if debug else ""}.json'
        )

        if os.path.exists(output_file) and not debug:
            print(f"\nSkipping {model_name} - results already exist in {output_file}")
            continue

        print(f"\nLoading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_info["path"], **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.eval()

        results = {}
        for dataset_name, dataset in datasets.items():
            print(f"\nEvaluating {model_name} on {dataset_name}...")
            results[dataset_name] = calculate_perplexities(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                context_length=model_info["context_length"],
            )

            # Clean up GPU memory after each dataset
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Save results for this model to a JSON file
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        # More thorough cleanup between models
        del model
        del tokenizer
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Force garbage collection
        import gc

        gc.collect()


if __name__ == "__main__":
    main()
