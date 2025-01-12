import math
import numpy as np
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import psutil
import torch.cuda
import json
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

def get_total_gpu_memory():
    """Helper function to get total memory usage across all devices."""
    if torch.cuda.is_available():
        return sum(
            torch.cuda.memory_allocated(d) for d in range(torch.cuda.device_count())
        )
    return psutil.Process().memory_info().rss


def print_performance_metrics(metrics):
    """Helper function to print performance metrics."""
    print("\nPerformance Metrics:")
    print(f"Tokens processed: {metrics['total_tokens']:,}")
    print(f"Inference time: {metrics['inference_time']:.2f} s")
    print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
    print(f"Peak memory usage: {metrics['peak_memory']/1024**2:.2f} MB")
    print(f"Memory efficiency: {metrics['tokens_per_mb']:.2f} tokens/MB")

def calculate_word_level_perplexity_v2(
    model, tokenizer, context_length, text, device="cuda"
):
    """
    Computes word-level metrics including:
    1. Word-level perplexity
    2. Token-level perplexity
    3. Cumulative Word-Level Bits (CWB)

    Word-level perplexity = exp(-1/W ∑ log P(word_j))
    where log P(word_j) = ∑_{k=1}^{M_j} log P(t_{j,k})

    Cumulative Word Bits (CWB):
    bits(word_j) = -log₂(P(word_j)) = ∑_{k=1}^{M_j} -log₂(P(t_{j,k}))
    CWB = (1/W) ∑_{j=1}^W bits(word_j)
    """
    encoded = tokenizer(
        text,
        max_length=context_length,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    offsets = encoded.offset_mapping[0].tolist()
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

    # Convert offsets directly to words by slicing the text
    pieces = [text[start:end] for start, end in offsets if text[start:end]]

    def is_word_boundary(token):
        """Helper to determine if a token marks a word boundary."""
        return (token.startswith(" ") or 
                token.startswith("\n") or 
                # Common Dutch/English punctuation marks
                token in [".", ",", "!", "?", ";", ":", "(", ")", "[", "]", 
                         """, """, "'", '"', "—", "-", "–"])

    # Merge tokens into proper words
    words = []
    word_log_sums_cumulative = []
    word_token_counts = []
    current_word = ""
    current_word_log_sum_cumulative = 0.0
    current_word_token_count = 0

    # Skip first piece since it has no prediction
    for i, token in enumerate(pieces[1:]):
        if is_word_boundary(token):
            if current_word:
                words.append(current_word)
                word_log_sums_cumulative.append(current_word_log_sum_cumulative)
                word_token_counts.append(current_word_token_count)
                current_word = ""
                current_word_log_sum_cumulative = 0.0
                current_word_token_count = 0
        current_word += token
        current_word_log_sum_cumulative += seq_token_log_probs[i]
        current_word_token_count += 1

    # Add final word
    if current_word:
        words.append(current_word.lstrip())
        word_log_sums_cumulative.append(current_word_log_sum_cumulative)
        word_token_counts.append(current_word_token_count)

    mean_word_log_prob = torch.asarray(word_log_sums_cumulative).mean().item()
    if not math.isfinite(mean_word_log_prob):
        return float("nan"), float("nan"), 0, 0, float('nan'), 0

    word_level_ppl = math.exp(-mean_word_log_prob)
    tokens_per_word = num_tokens / len(words)

    # Convert natural log to log2 and sum bits for each word
    word_bits = [-log_sum * math.log2(math.e) for log_sum in word_log_sums_cumulative]

    # Get token strings from ids for debugging
    # token_strings = [tokenizer.convert_ids_to_tokens(input_ids[0][i].item()) for i in range(1, len(input_ids[0]))]
    #
    # # Track cumulative token index
    # current_token_idx = 0
    # word_details = []
    # for word, count, bits in zip(words, word_token_counts, word_bits):
    #     word_details.append((word,
    #                        count,
    #                        bits.item(),
    #                        token_strings[current_token_idx:current_token_idx + count]))
    #     current_token_idx += count
        
    # print("\nWord details (word, token count, bits, tokens):")
    # for detail in word_details:
    #     print(detail)

    # Calculate CWB (average bits per word)
    mean_bits = torch.mean(torch.asarray(word_bits)).item()

    # Compare perplexities using relative difference
    # relative_diff = abs(word_level_ppl - token_perplexity) / min(word_level_ppl, token_perplexity)
    # if relative_diff > 2.0:  # Triggers if one is more than 3x the other
    #     print(f"\033[93mLarge perplexity difference detected:")
    #     print(f"Word perplexity: {word_level_ppl:.2f}")
    #     print(f"Token perplexity: {token_perplexity:.2f}")
    #     print(f"Relative difference: {relative_diff:.2f}x\033[0m")
    #     print(f"\033[93mText: {text[:200]}...\033[0m")  # Only show first 200 chars

    # if word_level_ppl > 1e6 or word_level_ppl < 1:
    #     print(f"Warning: Unusual perplexity value: {word_level_ppl}")

    return word_level_ppl, token_perplexity, tokens_per_word, num_tokens, mean_bits, len(words)


def calculate_perplexities(model, tokenizer, dataset, context_length):
    """
    Calculates perplexity metrics for a given model and dataset.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        dataset: The dataset to evaluate on.
        context_length: The context length to use.

    Returns:
        A dictionary containing the following keys:
        - dataset_name: The name of the dataset.
        - model_name: The name of the model.
        - aggregated_metrics: Aggregated metrics across all texts.
            - token_perplexity: Percentiles (25, 50, 75) of token perplexities.
            - word_perplexity: Percentiles (25, 50, 75) of word perplexities.
            - mean_token_perplexity: The mean token perplexity.
            - mean_word_perplexity: The mean word perplexity.
            - mean_tokens_per_word: The mean tokens per word.
            - mean_cumulative_word_bits: The mean cumulative word bits.
        - text_level_metrics: A list of dictionaries, each containing metrics for a single text.
            - text_id: The ID of the text.
            - token_perplexity: The token perplexity.
            - word_perplexity: The word perplexity.
            - tokens_per_word: The tokens per word.
            - cumulative_word_bits: The cumulative word bits.
    """
    token_ppls = []
    word_ppls = []
    tokens_per_word_list = []
    total_tokens = 0
    cumulative_word_bits_list = []
    detailed_metrics = []
    
    with tqdm(dataset, desc="Processing texts") as pbar:
        for i, row in enumerate(pbar):
            word_ppl, token_ppl, tpw, num_tokens, mean_bits, num_words = calculate_word_level_perplexity_v2(
                model, tokenizer, context_length, row["text"], device=DEVICE
            )
            
            detailed_metrics.append({
                "text_id": i,
                "token_perplexity": token_ppl,
                "word_perplexity": word_ppl,
                "tokens_per_word": tpw,
                "cumulative_word_bits": mean_bits,
                "num_tokens": num_tokens,
                "num_words": num_words
            })
            
            word_ppls.append(word_ppl)
            token_ppls.append(token_ppl)
            tokens_per_word_list.append(tpw)
            cumulative_word_bits_list.append(mean_bits)
            total_tokens += num_tokens
            
            pbar.set_postfix({
                "tokens": total_tokens,
                "token_ppl": f"{token_ppl:.1f}",
                "word_ppl": f"{word_ppl:.1f}",
                "tpw": f"{tpw:.2f}",
                "bits": f"{mean_bits:.1f}"
            })

    valid_token_ppls = [x for x in token_ppls if not (np.isnan(x) or np.isinf(x))]
    valid_word_ppls = [x for x in word_ppls if not (np.isnan(x) or np.isinf(x))]
    
    def tensor_to_float(x):
        """Helper function to safely convert tensor/scalar to float."""
        if torch.is_tensor(x):
            return float(x.float().cpu().numpy())  # Convert bfloat16 to float32 first
        return float(x)

    if valid_token_ppls and valid_word_ppls:
        # Move tensors to CPU before numpy operations
        token_percentiles = np.percentile([tensor_to_float(x) for x in valid_token_ppls], [25, 50, 75])
        word_percentiles = np.percentile([tensor_to_float(x) for x in valid_word_ppls], [25, 50, 75])
        mean_token_perplexity = np.mean([tensor_to_float(x) for x in valid_token_ppls])
        mean_word_perplexity = np.mean([tensor_to_float(x) for x in valid_word_ppls])
        mean_tokens_per_word = np.mean([tensor_to_float(x) for x in tokens_per_word_list])
        mean_cumulative_word_bits = np.mean([tensor_to_float(x) for x in cumulative_word_bits_list])
    else:
        token_percentiles = [float("nan")] * 3
        word_percentiles = [float("nan")] * 3
        mean_token_perplexity = float("nan")
        mean_word_perplexity = float("nan")
        mean_tokens_per_word = float("nan")
        mean_cumulative_word_bits = float("nan")

    return {
        "aggregated_metrics": {
            "token_perplexity": {
                "25": float(token_percentiles[0]),
                "50": float(token_percentiles[1]),
                "75": float(token_percentiles[2])
            },
            "word_perplexity": {
                "25": float(word_percentiles[0]),
                "50": float(word_percentiles[1]),
                "75": float(word_percentiles[2])
            },
            "mean_token_perplexity": float(mean_token_perplexity),
            "mean_word_perplexity": float(mean_word_perplexity),
            "mean_tokens_per_word": float(mean_tokens_per_word),
            "mean_cumulative_word_bits": float(mean_cumulative_word_bits)
        },
        "text_level_metrics": [
            {
                "text_id": m["text_id"],
                "token_perplexity": tensor_to_float(m["token_perplexity"]),
                "word_perplexity": tensor_to_float(m["word_perplexity"]),
                "tokens_per_word": tensor_to_float(m["tokens_per_word"]),
                "cumulative_word_bits": tensor_to_float(m["cumulative_word_bits"]),
                "num_tokens": m["num_tokens"],
                "num_words": m["num_words"]
            }
            for m in detailed_metrics
        ]
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
