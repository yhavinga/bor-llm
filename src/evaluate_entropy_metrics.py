import argparse
import json
import os

import numpy as np
import plotext as plt
import torch
import torch.cuda
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_cross_entropy_metrics(
    model, tokenizer, context_length, text, device="cuda", add_start_token=False
):
    """
    Computes sequence-level metrics:
    1. Token-level perplexity = exp(-log_likelihood / num_tokens)
    2. Bits per Word (BPW) = -log_likelihood_in_bits / num_words

    Where log_likelihood = ∑log P(token_i|token_{<i}, θ)
    θ represents model parameters

    Note: Model returns log probabilities in base e (from log_softmax),
    which we convert to base 2 for BPW calculation.
    """
    encoded = tokenizer(
        text,
        max_length=context_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    if add_start_token and tokenizer.bos_token_id is not None:
        bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)
        attention_mask = torch.cat([torch.ones_like(bos_tensor), attention_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Calculate token-level metrics using log_softmax
    next_token_logits = logits[:, :-1, :]  # (batch_size, seq_len - 1, vocab_size)
    targets = input_ids[:, 1:].to(device)  # (batch_size, seq_len - 1)
    shifted_attention_mask = attention_mask[:, 1:].to(
        device
    )  # (batch_size, seq_len - 1)

    # Number of predicted tokens (excluding padding)
    num_tokens = shifted_attention_mask.sum().item()

    # # Get conditional log probabilities for each token
    # conditional_log_probs = torch.log_softmax(next_token_logits, dim=-1)  # (batch_size, seq_len - 1, vocab_size)
    # conditional_token_log_probs = conditional_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # # Apply mask and compute sequence log-likelihood ∑log P(token_i|context)
    # valid_conditional_log_probs = conditional_token_log_probs * shifted_attention_mask[0]
    # neg_log_likelihood = -valid_conditional_log_probs.sum()
    # # Token-level perplexity from log-likelihood normalized by sequence length in #tokens
    # token_perplexity = torch.exp(neg_log_likelihood / num_tokens).item()

    # Using torch's crossentropyloss to do te same
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    neg_log_likelihood = loss_fct(next_token_logits.transpose(1, 2), targets)
    token_perplexity = torch.exp(neg_log_likelihood / num_tokens).item()

    # Convert log-likelihood to bits and normalize by sequence length in #words for BPW
    neg_log_likelihood_in_bits = neg_log_likelihood / torch.log(torch.tensor(2.0))
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    num_words = len(decoded_text.split())
    bits_per_word = (neg_log_likelihood_in_bits / num_words).item()

    return token_perplexity, num_tokens, bits_per_word, num_words


def plot_metrics(detailed_metrics, context_length):
    """
    Creates a terminal plot of perplexity and bits-per-word metrics.
    """
    plt.clf()
    # Get terminal size and set plot height
    term_size = plt.terminal_size()
    plot_height = max(
        12, int(term_size[1] * 0.75)
    )  # 75% of terminal height, minimum 12 rows
    plt.plotsize(None, plot_height)  # None preserves automatic width scaling

    token_ppls = [m["token_perplexity"] for m in detailed_metrics]
    bits_per_word_list = [m["bits_per_word"] for m in detailed_metrics]
    indices = list(range(len(token_ppls)))
    plt.plot(indices, token_ppls, label="Perplexity", color="red")
    plt.plot(indices, bits_per_word_list, label="Bits per word", color="blue")
    plt.title(f"Context length: {context_length}")
    plt.xlabel("Example index")
    plt.ylabel("Score")
    plt.show()


def calculate_dataset_metrics(model, tokenizer, dataset, context_length):
    """
    Calculates perplexity and bits-per-word metrics across a dataset
    """
    detailed_metrics = []
    running_ppl_sum = 0
    running_bpw_sum = 0

    with tqdm(dataset, desc="Processing texts") as pbar:
        for i, row in enumerate(pbar):
            (
                token_ppl,
                num_tokens,
                bits_per_word,
                num_words,
            ) = calculate_cross_entropy_metrics(
                model, tokenizer, context_length, row["text"], device=DEVICE
            )

            # Update running sums for valid metrics only
            if not (np.isnan(token_ppl) or np.isinf(token_ppl)):
                running_ppl_sum += token_ppl
                running_bpw_sum += bits_per_word
                mean_ppl = running_ppl_sum / (i + 1)
                mean_bpw = running_bpw_sum / (i + 1)

            detailed_metrics.append(
                {
                    "text_id": i,
                    "text": row["text"],
                    "token_perplexity": token_ppl,
                    "bits_per_word": bits_per_word,
                    "num_tokens": num_tokens,
                    "num_words": num_words,
                }
            )

            pbar.set_postfix(
                {
                    "token_ppl": f"{token_ppl:.1f}",
                    "bpw": f"{bits_per_word:.1f}",
                    "mean_ppl": f"{mean_ppl:.1f}",
                    "mean_bpw": f"{mean_bpw:.1f}",
                }
            )

    plot_metrics(detailed_metrics, context_length)

    valid_token_ppls = [
        x
        for x in [m["token_perplexity"] for m in detailed_metrics]
        if not (np.isnan(x) or np.isinf(x))
    ]
    valid_bpw = [
        x
        for x in [m["bits_per_word"] for m in detailed_metrics]
        if not (np.isnan(x) or np.isinf(x))
    ]

    def tensor_to_float(x):
        if torch.is_tensor(x):
            return float(x.float().cpu().numpy())
        return float(x)

    return {
        "model_info": {
            "context_length": context_length,
        },
        "aggregated_metrics": {
            "mean_perplexity": float(
                np.mean([tensor_to_float(x) for x in valid_token_ppls])
            ),
            "median_perplexity": float(
                np.median([tensor_to_float(x) for x in valid_token_ppls])
            ),
            "mean_bits_per_word": float(
                np.mean([tensor_to_float(x) for x in valid_bpw])
            ),
            "median_bits_per_word": float(
                np.median([tensor_to_float(x) for x in valid_bpw])
            ),
        },
        "text_level_metrics": sorted(
            [
                {
                    "text_id": m["text_id"],
                    "text": m["text"],
                    "perplexity": tensor_to_float(m["token_perplexity"]),
                    "bits_per_word": tensor_to_float(m["bits_per_word"]),
                    "num_tokens": m["num_tokens"],
                    "num_words": m["num_words"],
                }
                for m in detailed_metrics
            ],
            key=lambda x: x["bits_per_word"],
        ),
    }


def main(debug=False):
    print("Loading datasets...")

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
        tokenizer = AutoTokenizer.from_pretrained(
            model_info["path"], add_eos_token=True
        )
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.eval()

        results = {}
        for dataset_name, dataset in datasets.items():
            print(f"\nEvaluating {model_name} on {dataset_name}...")
            results[dataset_name] = calculate_dataset_metrics(
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with reduced dataset"
    )
    args = parser.parse_args()

    main(debug=args.debug)
