import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Define colors globally
colors = ["#4361EE", "#2EC4B6", "#7209B7", "#F72585"]

def load_evaluation_results(filename):
    """Load and parse evaluation results from JSON file."""
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def create_perplexity_distribution_plot(results_dict, output_dir):
    """Create violin plots showing perplexity and CWB distributions across models and datasets."""
    model_data = []
    
    # Process each model's results
    for model_name, results in results_dict.items():
        for dataset_name, data in results.items():
            if dataset_name in ['culturax_nl', 'mc4_nl', 'fineweb_edu_4']:
                if not isinstance(data, dict):
                    continue
                    
                # Filter token perplexities
                token_ppls = data.get('token_perplexities', [])
                
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
                model_data.extend([
                    {'model': model_name,
                     'dataset': dataset_name,
                     'metric': 'Token PPL',
                     'value': v}
                    for v in token_ppls
                ])
                
                # Filter word perplexities
                word_ppls = data.get('word_perplexities', [])
                word_ppls = remove_outliers(word_ppls)
                model_data.extend([
                    {'model': model_name,
                     'dataset': dataset_name, 
                     'metric': 'Word PPL',
                     'value': v}
                    for v in word_ppls
                ])
                
                # Add cumulative word bits
                word_bits = data.get('cumulative_word_bits', [])
                word_bits = remove_outliers(word_bits)
                model_data.extend([
                    {'model': model_name,
                     'dataset': dataset_name,
                     'metric': 'Word Bits',
                     'value': v}
                    for v in word_bits
                ])

                # Add tokens per word
                tokens_per_word = data.get('tokens_per_word', [])
                # tokens_per_word = remove_outliers(tokens_per_word)
                model_data.extend([
                    {'model': model_name,
                     'dataset': dataset_name,
                     'metric': 'Tokens per Word',
                     'value': v}
                    for v in tokens_per_word
                ])
    
    df = pd.DataFrame(model_data)

    # Define model order
    model_order = [
        'gpt-neo-125M-dutch',
        'gpt2-medium-dutch',
        'gpt2-large-dutch',
        'Llama-3.2-1B',
        'Bor-1B',
        'gpt-neo-1.3B-dutch',
        'Fietje-2',
        'Phi-3.5-mini-instruct'
    ]

    # Define dataset order (top to bottom in plot)
    dataset_order = ['mc4_nl', 'culturax_nl', 'fineweb_edu_4']
    
    # Set categorical ordering for both model and dataset
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)
    
    # Update plot style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(dataset_order), 1, figsize=(18, 18), height_ratios=[1]*len(dataset_order))
   
    
    # Dataset names mapping
    dataset_names = {
        'mc4_nl': 'MC4 NL Cleaned',
        'culturax_nl': 'CulturaX NL',
        'fineweb_edu_4': 'Fineweb Edu'
    }
    
    # Plot for each dataset
    for idx, dataset in enumerate(dataset_order):
        ax = axes[idx]
        dataset_data = df[df['dataset'] == dataset]
        
        # Create violin plots for perplexity metrics (left y-axis)
        ppl_data = dataset_data[dataset_data['metric'].isin(['Token PPL', 'Word PPL'])]
        sns.violinplot(data=ppl_data, x='model', y='value', hue='metric',
                      ax=ax, palette=colors[:2], split=True)
        ax.set_yscale('log')
        ax.set_ylabel('Perplexity (log scale)')
        
        # Create second y-axis for Word Bits and Tokens per Word
        ax2 = ax.twinx()
        word_bits_data = dataset_data[dataset_data['metric'] == 'Word Bits']
        
        # Plot Word Bits on ax2
        if not word_bits_data.empty:
            sns.violinplot(data=word_bits_data, x='model', y='value',
                         ax=ax2, color=colors[2], alpha=0.5)
            
            # Scale Word Bits
            word_bits_min = word_bits_data['value'].min()
            word_bits_max = word_bits_data['value'].max()
            if np.isfinite(word_bits_min) and np.isfinite(word_bits_max):
                margin = (word_bits_max - word_bits_min) * 0.1
                ax2.set_ylim(word_bits_min - margin, word_bits_max + margin)
            
            ax2.set_ylabel('Cumulative Word Bits', color=colors[2])
            ax2.tick_params(axis='y', labelcolor=colors[2])
        
        # Third y-axis for Tokens per Word
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # Plot Tokens per Word on ax3
        tokens_per_word_data = dataset_data[dataset_data['metric'] == 'Tokens per Word']
        if not tokens_per_word_data.empty:
            sns.violinplot(data=tokens_per_word_data, x='model', y='value',
                         ax=ax3, color=colors[3], alpha=0.5)
            
            # Scale Tokens per Word
            tokens_min = tokens_per_word_data['value'].min()
            tokens_max = tokens_per_word_data['value'].max()
            if np.isfinite(tokens_min) and np.isfinite(tokens_max):
                margin = (tokens_max - tokens_min) * 0.1
                ax3.set_ylim(tokens_min - margin, tokens_max + margin)
            
            ax3.set_ylabel('Tokens per Word', color=colors[3])
            ax3.tick_params(axis='y', labelcolor=colors[3])

        # Set title and adjust layout
        ax.set_title(dataset_names.get(dataset, dataset))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Handle legend
        if idx == 0:
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, _ = ax2.get_legend_handles_labels()
            handles3, _ = ax3.get_legend_handles_labels()
            
            # Create custom legend handles for Word Bits and Tokens per Word
            from matplotlib.patches import Patch
            handles2.append(Patch(color=colors[2], alpha=0.5, label='Word Bits'))
            handles3.append(Patch(color=colors[3], alpha=0.5, label='Tokens per Word'))
            
            all_handles = handles1 + handles2 + handles3
            all_labels = labels1 + ['Word Bits', 'Tokens per Word']
            
            if ax.get_legend():
                ax.get_legend().remove()
            if ax2.get_legend():
                ax2.get_legend().remove()
            if ax3.get_legend():
                ax3.get_legend().remove()
            
            ax.legend(all_handles, all_labels,
                     bbox_to_anchor=(1.15, 1), loc='upper left')
        else:
            if ax.get_legend():
                ax.get_legend().remove()
            if ax2.get_legend():
                ax2.get_legend().remove()
            if ax3.get_legend():
                ax3.get_legend().remove()
    
    plt.suptitle('Token PPL, Word PPL, Cumulative Word Bits, and Tokens per Word Distribution Across Models',
                y=1.02, fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/perplexity_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_comparison(results_dict, output_dir):
    """Create plot comparing performance metrics across models using seaborn."""
    metrics = []
    
    model_order = [
        'gpt-neo-125M-dutch',
        'gpt2-medium-dutch',
        'gpt2-large-dutch',
        'Llama-3.2-1B',
        'Bor-1B',
        'gpt-neo-1.3B-dutch',
        'Fietje-2',
        'Phi-3.5-mini-instruct'
    ]
    
    # Collect metrics
    for model_name, model_results in results_dict.items():
        if 'performance' in model_results:
            perf = model_results['performance']
            metrics.extend([
                {'model': model_name, 'metric': 'Throughput', 
                 'value': perf.get('throughput', 0)},
                {'model': model_name, 'metric': 'Memory Efficiency', 
                 'value': perf.get('tokens_per_mb', 0)}
            ])
        else:
            # Iterate over datasets to find performance data
            for dataset_name, dataset_results in model_results.items():
                if 'performance' in dataset_results:
                    perf = dataset_results['performance']
                    metrics.extend([
                        {'model': model_name, 'metric': 'Throughput', 
                         'value': perf.get('throughput', 0)},
                        {'model': model_name, 'metric': 'Memory Efficiency', 
                         'value': perf.get('tokens_per_mb', 0)}
                    ])
                    break  # Assume performance metrics are the same across datasets, so break after first find
    
    # Create DataFrame and set categorical order
    df = pd.DataFrame(metrics)
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    
    # Set style and color palette
    sns.set_style("whitegrid")
    
    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Throughput
    sns.barplot(
        data=df[df['metric'] == 'Throughput'],
        x='model',
        y='value',
        color=colors[0],
        ax=ax1
    )
    ax1.set_ylabel('Throughput (Tokens/second)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot Memory Efficiency
    sns.barplot(
        data=df[df['metric'] == 'Memory Efficiency'],
        x='model',
        y='value',
        color=colors[1],
        ax=ax2
    )
    ax2.set_ylabel('Memory Efficiency (Tokens/MB)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Set title for the entire figure
    fig.suptitle('Model Speed and Memory Efficiency', y=1.02, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    output_dir = 'docs/model-cards'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Collect all results first
    results_dict = {}
    for filename in glob.glob('*_evaluation_results.json'):
        model_name = filename.replace('_evaluation_results.json', '')
        results = load_evaluation_results(filename)
        results_dict[model_name] = results
    
    # Generate visualizations with complete data
    create_perplexity_distribution_plot(results_dict, output_dir)
    create_performance_comparison(results_dict, output_dir)

if __name__ == '__main__':
    main()