import glob
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define colors globally
colors = ["#4361EE", "#2EC4B6", "#7209B7", "#F72585"]


def load_evaluation_results(filename):
    """Load and parse evaluation results from JSON file."""
    with open(filename, "r") as f:
        results = json.load(f)
    return results


def create_perplexity_distribution_plot(results_dict, output_dir):
    """Create side-by-side violin plots for token PPL and BPW."""
    model_data = []

    # Process each model's results but only for culturax and fineweb
    for model_name, results in results_dict.items():
        for dataset_name, data in results.items():
            if dataset_name in [
                "culturax_nl",
                "fineweb_edu_4",
            ]:  # Only these two datasets
                if not isinstance(data, dict):
                    continue

                token_ppls = [
                    text_data["perplexity"] for text_data in data["text_level_metrics"]
                ]
                bits_per_word_list = [
                    text_data["bits_per_word"]
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
                bits_per_word_list = remove_outliers(bits_per_word_list)

                model_dataset = f"{model_name} ({dataset_name})"

                model_data.extend(
                    [
                        {
                            "model": model_name,
                            "model_dataset": model_dataset,
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
                            "model_dataset": model_dataset,
                            "dataset": dataset_name,
                            "metric": "BPW",
                            "value": v,
                        }
                        for v in bits_per_word_list
                    ]
                )

    df = pd.DataFrame(model_data)

    model_order = [
        "gpt-neo-125M-dutch",
        "gpt2-medium-dutch",
        "gpt2-large-dutch",
        "Bor-1B",
        "Llama-3.2-1B",
        # "gpt-neo-1.3B-dutch",
        "EuroLLM-1.7B",
        "SmolLM2-1.7B",
        "Fietje-2",
        "Phi-3.5-mini-instruct",
        "Boreas-7B",
        "Phi-4",
    ]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Get HUSL color palette - one color per model
    colors = sns.color_palette("husl", n_colors=len(model_order))

    # Plot Token PPL with split violins
    sns.violinplot(
        data=df[df["metric"] == "Token PPL"],
        x="value",
        y="model",
        hue="dataset",
        ax=ax1,
        split=True,
        density_norm="count",
        inner="box",
        order=model_order,
        hue_order=["culturax_nl", "fineweb_edu_4"],
    )

    # Plot BPW with split violins
    sns.violinplot(
        data=df[df["metric"] == "BPW"],
        x="value",
        y="model",
        hue="dataset",
        ax=ax2,
        split=True,
        density_norm="count",
        inner="box",
        order=model_order,
        hue_order=["culturax_nl", "fineweb_edu_4"],
    )

    # Apply colors to violin plots - ensuring consistent colors
    for ax in [ax1, ax2]:
        for idx, violin in enumerate(
            ax.collections[::2]
        ):  # Step by 2 because split violins create pairs
            color_idx = len(model_order) - 1 - idx  # Reverse index to match plot order
            violin.set_facecolor(colors[color_idx])
            violin.set_alpha(0.7)  # Higher alpha for CulturaX
            ax.collections[idx * 2 + 1].set_facecolor(colors[color_idx])
            ax.collections[idx * 2 + 1].set_alpha(0.4)  # Lower alpha for Fineweb

    ax1.set_xlabel("Token Perplexity")
    ax1.set_ylabel("Models")
    ax1.grid(True, linestyle="--", alpha=0.7)

    for ax in [ax1, ax2]:
        ax.get_legend().remove()
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="gray", alpha=0.8, label="CulturaX (Dutch)"),
            Patch(facecolor="gray", alpha=0.3, label="Fineweb (Eng)"),
        ]
        ax.legend(
            handles=legend_elements,
            title="",
            loc="upper right",
            frameon=True,
            handlelength=1.5,
            handleheight=1.5,
        )

    ax2.set_xlabel("Bits Per Word")
    ax2.set_ylabel("")
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle(
        "Token Perplexity and Bits Per Word Distribution Across Models\nBy Dataset",
        y=1.02,
        fontsize=16,
    )
    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/perplexity_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    input_dir = "outputs"
    output_dir = "docs/model-cards"
    Path(output_dir).mkdir(exist_ok=True)

    # Collect all results first
    results_dict = {}
    for filename in glob.glob(f"{input_dir}/*_evaluation_results.json"):
        model_name = Path(filename).stem.replace("_evaluation_results", "")
        results = load_evaluation_results(filename)
        results_dict[model_name] = results

    create_perplexity_distribution_plot(results_dict, output_dir)


if __name__ == "__main__":
    main()
