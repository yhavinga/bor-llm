import glob
import json
import pandas as pd
import numpy as np

MODEL_PARAMS = {
    'gpt-neo-125M-dutch': 176_000_000,
    'gpt2-medium-dutch': 380_000_000,
    'gpt2-large-dutch': 812_000_000,
    'Bor-1B': 1_190_000_000,
    'Llama-3.2-1B': 1_240_000_000,
    'gpt-neo-1.3B-dutch': 1_420_000_000,
    'Fietje-2': 2_700_000_000,
    'Phi-3.5-mini-instruct': 3_800_000_000
}

MODEL_LANGUAGE = {
    'gpt-neo-125M-dutch': '🇳🇱',      # Dutch only
    'gpt2-medium-dutch': '🇳🇱',       # Dutch only
    'gpt2-large-dutch': '🇳🇱',        # Dutch only
    'gpt-neo-1.3B-dutch': '🇳🇱',      # Dutch only
    'Bor-1B': '🇳🇱🇬🇧',              # Dutch + English
    'Llama-3.2-1B': '🌐',             # Multilingual
    'Fietje-2': '🇳🇱*',               # Dutch-tuned Phi
    'Phi-3.5-mini-instruct': '🌐'     # Multilingual
}

MODEL_CONTEXT_LENGTH = {
    'gpt-neo-125M-dutch': 512,
    'gpt2-medium-dutch': 512,
    'gpt2-large-dutch': 512,
    'gpt-neo-1.3B-dutch': 512,
    'Bor-1B': 4096,
    'Llama-3.2-1B': 128_000,
    'Fietje-2': 2048,
    'Phi-3.5-mini-instruct': 128_000
}

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def compute_averages():
    results = []
    
    # Define dataset order
    dataset_order = ['mc4_nl', 'culturax_nl', 'fineweb_edu_4']
    
    for filename in glob.glob('*_evaluation_results.json'):
        model_name = filename.replace('_evaluation_results.json', '')
        data = load_results(filename)
        
        for dataset_name, dataset_data in data.items():
            if dataset_name not in dataset_order:
                continue
                
            if not isinstance(dataset_data, dict) or 'text_level_metrics' not in dataset_data:
                continue
                
            metrics = dataset_data['text_level_metrics']
            
            avg_metrics = {
                'token_perplexity': np.mean([m['token_perplexity'] for m in metrics]),
                'tokens_per_word': np.mean([m['tokens_per_word'] for m in metrics]),
                'cumulative_word_bits': np.mean([m['cumulative_word_bits'] for m in metrics])
            }
            
            results.append({
                'model': model_name,
                'dataset': dataset_name,
                **avg_metrics
            })
    
    df = pd.DataFrame(results)
    df = df.round(2)
    
    # Print model parameters table first
    print("\n### Model Parameters\n")
    params_df = pd.DataFrame([MODEL_PARAMS]).T.reset_index()
    params_df.columns = ['Model', 'Parameters']
    params_df['Language'] = params_df['Model'].map(MODEL_LANGUAGE)
    params_df['Context Length'] = params_df['Model'].map(MODEL_CONTEXT_LENGTH)
    params_df = params_df.sort_values('Parameters')
    print(params_df.to_markdown(index=False))
    
    # Create pivot tables with correct index (datasets only)
    metrics = ['token_perplexity', 'tokens_per_word', 'cumulative_word_bits']
    for metric in metrics:
        print(f"\n### {metric}\n")
        pivot_df = df.pivot(index='dataset', columns='model', values=metric)
        
        # Sort columns by model size
        model_sizes = {col: MODEL_PARAMS.get(col, 0) for col in pivot_df.columns}
        sorted_columns = sorted(pivot_df.columns, key=lambda x: model_sizes[x])
        pivot_df = pivot_df[sorted_columns]
        pivot_df = pivot_df.reindex(dataset_order)
        
        # No more flag addition to column names
        print(pivot_df.round(2).to_markdown())

if __name__ == '__main__':
    compute_averages()

"""
### Model Parameters

| Model                 |   Parameters | Language   |   Context Length |
|:----------------------|-------------:|:-----------|-----------------:|
| gpt-neo-125M-dutch    |    176000000 | 🇳🇱         |              512 |
| gpt2-medium-dutch     |    380000000 | 🇳🇱         |              512 |
| gpt2-large-dutch      |    812000000 | 🇳🇱         |              512 |
| Bor-1B                |   1190000000 | 🇳🇱🇬🇧       |             4096 |
| Llama-3.2-1B          |   1240000000 | 🌐          |           128000 |
| gpt-neo-1.3B-dutch    |   1420000000 | 🇳🇱         |              512 |
| Fietje-2              |   2700000000 | 🇳🇱*        |             2048 |
| Phi-3.5-mini-instruct |   3800000000 | 🌐          |           128000 |

### token_perplexity

| dataset       |   gpt-neo-125M-dutch |   gpt2-medium-dutch |   gpt2-large-dutch |   Bor-1B |   Llama-3.2-1B |   gpt-neo-1.3B-dutch |   Fietje-2 |   Phi-3.5-mini-instruct |
|:--------------|---------------------:|--------------------:|-------------------:|---------:|---------------:|---------------------:|-----------:|------------------------:|
| mc4_nl        |                23.2  |               16.62 |              18.27 |    15.1  |          13.33 |                17.93 |       4.62 |                   10.62 |
| culturax_nl   |                29.83 |               21.18 |              23.45 |    16.18 |          14.31 |                22.89 |       4.66 |                   11.24 |
| fineweb_edu_4 |                28.51 |               17.5  |              18.89 |    14.69 |          11    |                21.27 |      12.92 |                    6.86 |

### tokens_per_word

| dataset       |   gpt-neo-125M-dutch |   gpt2-medium-dutch |   gpt2-large-dutch |   Bor-1B |   Llama-3.2-1B |   gpt-neo-1.3B-dutch |   Fietje-2 |   Phi-3.5-mini-instruct |
|:--------------|---------------------:|--------------------:|-------------------:|---------:|---------------:|---------------------:|-----------:|------------------------:|
| mc4_nl        |                 1.28 |                1.28 |               1.28 |     1.37 |           1.76 |                 1.28 |       2.08 |                    1.84 |
| culturax_nl   |                 1.31 |                1.31 |               1.31 |     1.4  |           1.78 |                 1.31 |       2.11 |                    1.87 |
| fineweb_edu_4 |                 1.72 |                1.72 |               1.72 |     1.3  |           1.18 |                 1.72 |       1.16 |                    1.31 |

### cumulative_word_bits

| dataset       |   gpt-neo-125M-dutch |   gpt2-medium-dutch |   gpt2-large-dutch |   Bor-1B |   Llama-3.2-1B |   gpt-neo-1.3B-dutch |   Fietje-2 |   Phi-3.5-mini-instruct |
|:--------------|---------------------:|--------------------:|-------------------:|---------:|---------------:|---------------------:|-----------:|------------------------:|
| mc4_nl        |                 5.24 |                4.66 |               4.8  |     5.01 |           6.23 |                 4.76 |       4.31 |                    5.93 |
| culturax_nl   |                 6.04 |                5.39 |               5.56 |     5.28 |           6.46 |                 5.54 |       4.42 |                    6.18 |
| fineweb_edu_4 |                 8.18 |                6.97 |               7.14 |     4.89 |           3.95 |                 7.46 |       4.13 |                    3.48 |
"""