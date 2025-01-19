import gc
import json
import os

import evaluate
import torch
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(model_name, model_info, datasets, debug=False):
    """Evaluate a single model on multiple datasets."""
    print(f"\nEvaluating {model_name}...")

    results = {}
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        
        # Create fresh metric instances for each dataset
        perplexity_metric = evaluate.load("perplexity")
        bpw_metric = evaluate.load("bits_per_word.py")
        
        perplexity_metric.add_batch(predictions=dataset["text"])
        bpw_metric.add_batch(predictions=dataset["text"])

        perplexity_results = perplexity_metric.compute(
            model_id=model_info["path"],
            max_length=model_info["context_length"],
            add_start_token=False,
        )
        
        bpw_results = bpw_metric.compute(
            model_id=model_info["path"],
            max_length=model_info["context_length"],
            add_start_token=False,
        )

        results[dataset_name] = {
            "aggregated_metrics": {
                "mean_perplexity": float(torch.mean(torch.tensor(perplexity_results["perplexities"])).item()),
                "median_perplexity": float(torch.median(torch.tensor(perplexity_results["perplexities"])).item()),
                "mean_bits_per_word": float(torch.mean(torch.tensor(bpw_results["bits_per_word_scores"])).item()),
                "median_bits_per_word": float(torch.median(torch.tensor(bpw_results["bits_per_word_scores"])).item()),
            },
            "text_level_metrics": sorted([
                {
                    "text_id": i,
                    "text": dataset["text"][i],
                    "perplexity": float(ppl),
                    "bits_per_word": float(bpw),
                }
                for i, (ppl, bpw) in enumerate(zip(perplexity_results["perplexities"], bpw_results["bits_per_word_scores"]))
            ], key=lambda x: x["bits_per_word"])
        }

        # Clean up metric instances
        del perplexity_metric
        del bpw_metric

    # Cleanup
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()

    return results


def main():
    debug = True

    if debug:
        datasets = {
            "culturax_nl": load_dataset(
                "yhavinga/culturax_dutch_test", split="train[:73]", num_proc=16
            ),
        }
        models = {
            "gpt-neo-125M-dutch": {
                "path": "yhavinga/gpt-neo-125M-dutch",
                "context_length": 512,
            }
        }
    else:
        datasets = {
            "fineweb_edu_4": load_dataset(
                "yhavinga/fineweb_edu_score_gt_4_test", split="test", num_proc=16
            ),
            # "mc4_nl": load_dataset(
            #     "yhavinga/mc4_nl_cleaned",
            #     "tiny",
            #     split="validation[:1000]",
            #     num_proc=16,
            # ),
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

    for model_name, model_info in models.items():
        safe_model_name = model_name.replace("/", "_")
        output_file = (
            f'{safe_model_name}_evaluation_results{"_debug" if debug else ""}.json'
        )

        if os.path.exists(output_file) and not debug:
            print(f"\nSkipping {model_name} - results already exist in {output_file}")
            continue

        try:
            results = evaluate_model(model_name, model_info, datasets, debug)

            # Save results
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue


if __name__ == "__main__":
    main()
