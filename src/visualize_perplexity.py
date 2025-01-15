import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import math

# Define colors globally
colors = ["#4361EE", "#2EC4B6", "#7209B7", "#F72585"]


def load_evaluation_results(filename):
    """Load and parse evaluation results from JSON file."""
    with open(filename, "r") as f:
        results = json.load(f)
    return results


def create_perplexity_distribution_plot(results_dict, output_dir):
    """Create violin plots showing perplexity and CWB distributions across models and datasets."""
    model_data = []

    # Process each model's results
    for model_name, results in results_dict.items():
        for dataset_name, data in results.items():
            if dataset_name in ["culturax_nl", "mc4_nl", "fineweb_edu_4"]:
                if not isinstance(data, dict):
                    continue

                # Access token_perplexities and cumulative_word_bits from text_level_metrics
                token_ppls = [
                    text_data["token_perplexity"]
                    for text_data in data["text_level_metrics"]
                ]
                cumulative_word_bits = [
                    text_data["cumulative_word_bits"]
                    for text_data in data["text_level_metrics"]
                ]

                # Filter out extreme outliers (values beyond 3 IQR)
                def remove_outliers(x):
                    if len(x) == 0:
                        return x
                    q1 = np.percentile(x, 25)
                    q3 = np.percentile(x, 75)
                    iqr = q3 - q1
                    upper_bound = q3 + 3 * iqr
                    return [v for v in x if v <= upper_bound]

                token_ppls = remove_outliers(token_ppls)
                cumulative_word_bits = remove_outliers(cumulative_word_bits)

                model_data.extend(
                    [
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "metric": "Token PPL",
                            "value": v,
                        }
                        for v in token_ppls
                    ]
                )

                model_data.extend(
                    [
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "metric": "CWB",
                            "value": v,
                        }
                        for v in cumulative_word_bits
                    ]
                )

    df = pd.DataFrame(model_data)

    # Define model order
    model_order = [
        "gpt-neo-125M-dutch",
        "gpt2-medium-dutch",
        "gpt2-large-dutch",
        "Llama-3.2-1B",
        "Bor-1B",
        "gpt-neo-1.3B-dutch",
        "SmolLM2-1.7B",
        "Fietje-2",
        "Phi-3.5-mini-instruct",
    ]

    # Define dataset order (top to bottom in plot)
    dataset_order = ["mc4_nl", "culturax_nl", "fineweb_edu_4"]

    # Set categorical ordering for both model and dataset
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df["dataset"] = pd.Categorical(
        df["dataset"], categories=dataset_order, ordered=True
    )

    # Update plot style
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Create figure with subplots
    fig, axes = plt.subplots(
        len(dataset_order), 1, figsize=(18, 18), height_ratios=[1] * len(dataset_order)
    )

    # Dataset names mapping
    dataset_names = {
        "mc4_nl": "MC4 NL Cleaned",
        "culturax_nl": "CulturaX NL",
        "fineweb_edu_4": "Fineweb Edu",
    }

    # Plot for each dataset
    for idx, dataset in enumerate(dataset_order):
        ax = axes[idx]
        dataset_data = df[df["dataset"] == dataset]

        # Create violin plots for perplexity and CWB metrics
        sns.violinplot(
            data=dataset_data,
            x="model",
            y="value",
            hue="metric",
            ax=ax,
            palette=colors[:2],
            split=True,
            density_norm="count",
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)  # Set minimum y value to 1 (10^0)
        ax.set_ylabel("Value (log scale)")

        # Set title and adjust layout
        ax.set_title(dataset_names.get(dataset, dataset))
        ax.grid(True, linestyle="--", alpha=0.7)

        # Handle legend
        if idx == 0:
            handles1, labels1 = ax.get_legend_handles_labels()
            if ax.get_legend():
                ax.get_legend().remove()
            ax.legend(handles1, labels1, bbox_to_anchor=(1.15, 1), loc="upper left")
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.suptitle(
        "Token Perplexity and Cumulative Word Bits Distribution Across Models",
        y=1.02,
        fontsize=16,
    )
    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/perplexity_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_performance_comparison(results_dict, output_dir):
    """Create plot comparing performance metrics across models using seaborn."""
    # Skip performance comparison if no performance metrics are available
    if not results_dict or not any(
        "performance" in dataset_results
        for model_results in results_dict.values()
        for dataset_results in model_results.values()
    ):
        print(
            "No performance metrics found in results. Skipping performance comparison plot."
        )
        return

    metrics = []
    model_order = [
        "gpt-neo-125M-dutch",
        "gpt2-medium-dutch",
        "gpt2-large-dutch",
        "Llama-3.2-1B",
        "Bor-1B",
        "gpt-neo-1.3B-dutch",
        "SmolLM2-1.7B",
        "Fietje-2",
        "Phi-3.5-mini-instruct",
    ]

    # Collect metrics
    for model_name, model_results in results_dict.items():
        # Iterate over datasets to find performance data
        for dataset_name, dataset_results in model_results.items():
            if "performance" in dataset_results:
                perf = dataset_results["performance"]
                metrics.extend(
                    [
                        {
                            "model": model_name,
                            "metric": "Throughput",
                            "value": perf.get("throughput", 0),
                        },
                        {
                            "model": model_name,
                            "metric": "Memory Efficiency",
                            "value": perf.get("tokens_per_mb", 0),
                        },
                    ]
                )
                break  # Assume performance metrics are the same across datasets

    # Only create plot if we have metrics
    if not metrics:
        print(
            "No performance metrics found in results. Skipping performance comparison plot."
        )
        return

    # Create DataFrame and set categorical order
    df = pd.DataFrame(metrics)
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

    # Set style and color palette
    sns.set_style("whitegrid")

    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot Throughput
    sns.barplot(
        data=df[df["metric"] == "Throughput"],
        x="model",
        y="value",
        color=colors[0],
        ax=ax1,
    )
    ax1.set_ylabel("Throughput (Tokens/second)")
    ax1.tick_params(axis="x", rotation=45)

    # Plot Memory Efficiency
    sns.barplot(
        data=df[df["metric"] == "Memory Efficiency"],
        x="model",
        y="value",
        color=colors[1],
        ax=ax2,
    )
    ax2.set_ylabel("Memory Efficiency (Tokens/MB)")
    ax2.tick_params(axis="x", rotation=45)

    # Set title for the entire figure
    fig.suptitle("Model Speed and Memory Efficiency", y=1.02, fontsize=14)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    output_dir = "docs/model-cards"
    Path(output_dir).mkdir(exist_ok=True)

    # Collect all results first
    results_dict = {}
    for filename in glob.glob("*_evaluation_results.json"):
        model_name = filename.replace("_evaluation_results.json", "")
        results = load_evaluation_results(filename)
        results_dict[model_name] = results

    # Generate visualizations with complete data
    create_perplexity_distribution_plot(results_dict, output_dir)
    create_performance_comparison(results_dict, output_dir)


if __name__ == "__main__":
    main()
