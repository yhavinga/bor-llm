import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define colors globally
colors = ["#4361EE", "#2EC4B6"]

def load_evaluation_results(filename):
    """Load and parse evaluation results file."""
    with open(filename, 'r') as f:
        content = f.read()

    sections = content.split('\n\n')
    results = {}
    current_dataset = None
    
    for section in sections:
        if not section.strip():
            continue
            
        if 'Results:' in section:
            current_dataset = section.split(' Results:')[0].strip()
            results[current_dataset] = {'perplexities': [], 'metrics': {}}
            continue
            
        if current_dataset and 'Sample_ID\tToken_PPL\tWord_PPL' in section:
            # Parse perplexity data
            lines = [line for line in section.strip().split('\n') if line.strip()]
            data = []
            
            # Skip the header row
            for line in lines[1:]:
                if line.strip():
                    sample_id, token_ppl, word_ppl = line.split('\t')
                    try:
                        data.append({
                            'Sample_ID': int(sample_id),
                            'token_ppl': float(token_ppl),
                            'word_ppl': float(word_ppl) if word_ppl != 'N/A' else np.nan
                        })
                    except ValueError:
                        continue  # Skip any malformed lines
            
            if data:  # Only create DataFrame if we have valid data
                df = pd.DataFrame(data)
                results[current_dataset]['perplexities'] = df
            
        if 'Performance Metrics:' in section:
            metrics = {}
            for line in section.split('\n'):
                if ':' in line:
                    key, value = line.split(':')
                    try:
                        metrics[key.strip()] = float(value.split()[0])
                    except:
                        continue
            results['performance'] = metrics
            
    return results

def create_perplexity_distribution_plot(results_dict, output_dir):
    """Create violin plots showing perplexity distributions across models and datasets."""
    model_data = []
    
    # Process each model's results
    for model_name, results in results_dict.items():
        for dataset_name, data in results.items():
            if dataset_name in ['culturax_nl', 'mc4_nl']:
                if not isinstance(data, dict) or 'perplexities' not in data:
                    continue
                    
                df = data['perplexities']
                
                # Filter out extreme outliers (values beyond 3 IQR)
                def remove_outliers(x):
                    if len(x) == 0:
                        return x
                    q1 = np.percentile(x, 25)
                    q3 = np.percentile(x, 75)
                    iqr = q3 - q1
                    upper_bound = q3 + 3 * iqr
                    return [v for v in x if v <= upper_bound]
                
                # Filter token perplexities
                token_ppls = remove_outliers([v for v in df['token_ppl'] if not np.isnan(v)])
                model_data.extend([
                    {'model': model_name,
                     'dataset': dataset_name,
                     'metric': 'Token PPL',
                     'value': v}
                    for v in token_ppls
                ])
                
                # Filter word perplexities
                word_ppls = remove_outliers([v for v in df['word_ppl'] if not np.isnan(v)])
                model_data.extend([
                    {'model': model_name,
                     'dataset': dataset_name, 
                     'metric': 'Word PPL',
                     'value': v}
                    for v in word_ppls
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
        'Phi-3.5-mini-instruct'
    ]

    # Filter df to only include models in the order list
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    
    # Increase font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Create faceted plot with grid
    g = sns.FacetGrid(df, row='dataset', height=6, aspect=3)
    
    # Map the violin plot with gridlines
    def plot_violin_with_grid(data, **kwargs):
        ax = plt.gca()
        ax.grid(True, linestyle='--', alpha=0.7)
        return sns.violinplot(data=data, **kwargs)
    
    g.map_dataframe(plot_violin_with_grid,
        x='model',
        y='value',
        hue='metric',
        split=True,
        density_norm='width',
        palette=colors
    )
    
    # Define dataset names mapping
    dataset_names = {
        'mc4_nl': 'MC4 NL Cleaned',
        'culturax_nl': 'CulturaX NL'
    }
    
    # Update row titles directly
    for ax, title in zip(g.axes.flat, g.row_names):
        ax.set_title(dataset_names.get(title, title), size=14)
    
    g.set_axis_labels('Model', 'Perplexity (log scale)')
    
    # Set y-axis to log scale and rotate x-labels for all facets
    for ax in g.axes.flat:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add suptitle to the entire figure
    g.fig.suptitle('Token and Word Perplexity Distribution Across Models', y=1.02, fontsize=16)
    
    plt.tight_layout()
    
    # Save the plot with additional top margin for the suptitle
    plt.savefig(f'{output_dir}/perplexity_distribution.png', dpi=300, bbox_inches='tight')
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
        'Phi-3.5-mini-instruct'
    ]
    
    # Collect metrics
    for model_name in model_order:
        filename = f'{model_name}_evaluation_results.txt'
        if Path(filename).exists():
            model_results = load_evaluation_results(filename)
            if 'performance' in model_results:
                perf = model_results['performance']
                metrics.extend([
                    {'model': model_name, 'metric': 'Throughput', 
                     'value': perf.get('Throughput', 0)},
                    {'model': model_name, 'metric': 'Memory Efficiency', 
                     'value': perf.get('Memory Efficiency', 0)}
                ])
    
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
    for filename in glob.glob('*_evaluation_results.txt'):
        model_name = filename.replace('_evaluation_results.txt', '')
        results = load_evaluation_results(filename)
        results_dict[model_name] = results
    
    # Generate visualizations with complete data
    create_perplexity_distribution_plot(results_dict, output_dir)
    create_performance_comparison(results_dict, output_dir)

if __name__ == '__main__':
    main()