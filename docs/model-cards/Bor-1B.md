# Bor-1B - Dutch/English Pre-trained Language Model

## Model Description

<img src="bor-de-wolf.jpg" align="left" width="190" style="margin-right: 15px; margin-bottom: 10px;">

Bor-1B is a pre-trained multilingual language model based on the Mistral architecture, specifically designed for Dutch and English language processing. The model was pre-trained in two phases with different context lengths and dataset compositions.

The model's name pays homage to "Bor de Wolf" from the beloved Dutch children's series De Fabeltjeskrant (1968-1989). While the original Bor was famous for his signature "Huuuuu!" howl echoing through the forest, this Bor's howls are more subtle - processing Dutch and English text without disturbing the neighbors!

<div style="clear: both;"></div>

## Architecture
The Mistral architecture was chosen for its broad library support and efficient implementation. The Bor-1B model maintains Mistral's core structural elements while scaling down dimensions for a more compact design:

- **Model Scale**: ~1.19B parameters vs Mistral's 7B, primarily due to reduced width:
  - Hidden size: 1536 vs 4096
  - Intermediate MLP size: 6144 vs 14336
  - Attention heads remain at 32 with 8 KV heads (4:1 grouping ratio)
  - Layer depth maintained at 32 layers

- **Key Similarities**:
  - Sliding window attention (4096 tokens)
  - Vocabulary size (32K tokens)
  - Number of attention heads and KV heads
  - Model depth (32 layers)
  - Basic architecture components (RMSNorm, SiLU activation)

- **RoPE**:
  - A higher RoPE base frequency (θ = 80000 vs Mistral v0.1's 10000) was chosen to potentially improve extrapolation to sequences longer than those seen during training.

## Tokenizer
The model uses the [dutch-llama-tokenizer](https://huggingface.co/yhavinga/dutch-llama-tokenizer), a 32K BPE tokenizer specifically optimized for Dutch-English bilingual content and code. This tokenizer was selected for its optimal balance between:

1. Efficient tokenization of Dutch text (comparable to specialized Dutch tokenizers)
2. Strong performance on English and code
3. Compact vocabulary size (32K tokens) suitable for smaller models
4. Comprehensive character coverage including special characters for code and math

Benchmarks show it achieves tokenization efficiency within 5-10% of specialized single-language tokenizers while maintaining a vocabulary size appropriate for 1-2B parameter models. For mixed Dutch-English-code content, it typically requires 10-15% fewer tokens than general-purpose tokenizers like Mistral's.

## Training Data

The model was trained in two phases with different context lengths and dataset compositions:

![BOR Pretraining Dataset Composition](bor-1b-pretrain-datasets.png)

### Phase 1 (2048 context length)
Total tokens: 37.6B tokens × 3 epochs = 113B total
- Primary Dutch content (~82%): [MC4 Dutch Cleaned](https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned), Wikipedia,
 [GeminiPhiDutch](https://huggingface.co/datasets/Kalamazooter/GeminiPhiDutch) and additional Dutch datasets
- English content (~12%) from curated sources, including [goodwiki](https://huggingface.co/datasets/euirim/goodwiki)
- Python code (~6%) for programming capabilities: Python subsets from [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) and [github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean/viewer/Python-all)

### Phase 2 (4096 context length)
Total tokens: 37.6B tokens × 2.3 epochs = 87B total
- Balanced composition: Dutch (55%), English (34%), and specialized content (~11%)
- Dutch content: [MC4 Dutch Cleaned](https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned), Wikipedia,
 [GeminiPhiDutch](https://huggingface.co/datasets/Kalamazooter/GeminiPhiDutch), [CulturaX Dutch](https://huggingface.co/datasets/uonlp/CulturaX/viewer/nl)(*)
- English content: [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2), [goodwiki](https://huggingface.co/datasets/euirim/goodwiki) and additional curated sources
- Code and mathematics (~10%): Python from [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) and [github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean/viewer/Python-all), plus [FineWeb-math](https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-math-corpus) and [FineWeb-code](https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus)
- Instruction tuning (~2%): [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) (85%) and [OpenHermes 2.5 Dutch](https://huggingface.co/datasets/yhavinga/Openhermes-2.5-dutch-46k) (15%)

The second phase features a more balanced language distribution and enhanced code/mathematics representation to improve the model's capabilities across natural language and technical tasks.

## Training Details

### Hardware and Infrastructure
- Training hardware: TPU v4-32
- Training duration: 45 days (Nov 10 - Dec 24, 2023)

### Training Configuration
Common settings:
- Optimizer: AdamW (β1=0.92, β2=0.999, ε=1e-7)
- Weight decay: 0.1
- Z-loss: 0.0001
- Precision: bfloat16
- Attention implementation: sharded vanilla attention

Phase 1 (2048 context):
- Batch size: 128 (gradient accumulation: 2)
- Learning rate: 3e-4 → 5e-5 (linear decay)
- Warmup steps: 4000
- Training epochs: 2 (37B tokens × 3 epochs = 112B tokens)
- Max gradient norm: 0.7

Phase 2 (4096 context):
- Batch size: 32 (gradient accumulation: 8)
- Learning rate: 1.5e-4 → 3e-5 (linear decay)
- Warmup steps: 10000
- Training epochs: 1 (37B tokens × 2.4 epochs = 87B tokens)
- Max gradient norm: 1.0

### Training Dynamics
Training required multiple restarts due to software and infrastructure constraints, resulting in repeated learning rate warmups visible in the loss curves. Despite these interruptions, the model demonstrated stable convergence:

![Training Loss for Bor-1B](bor-1b-training-loss.png)

- Final training loss stabilized around 2.1-2.2 after initial optimization
- Phase 2 (4096 context) shows higher loss, most likely due to dataset composition:
  - More diverse content mix (34% English vs 12% in Phase 1, inclusion of CulturaX for Dutch)
  - More of technical content (mathematics, code)
- Multiple warmup cycles visible but maintained consistent loss levels

## Evaluation

The comparison models were selected based on:
1. Parameter count (ranging from 176M to 3.8B) to provide context around Bor-1B's size class
2. Dutch language capabilities (either Dutch-specific or strong multilingual models)


| Model                 | Parameters | Language | Context Length |
|:----------------------|----------:|:---------|--------------:|
| gpt-neo-125M-dutch    | 176M      | Dutch    | 512           |
| gpt2-medium-dutch     | 380M      | Dutch    | 512           |
| gpt2-large-dutch      | 812M      | Dutch    | 512           |
| Bor-1B                | 1.19B     | Dutch/English mixed | 4096          |
| Llama-3.2-1B          | 1.24B     | Multilingual | 128000        |
| gpt-neo-1.3B-dutch    | 1.42B     | Dutch    | 512           |
| SmolLM2-1.7B          | 1.7B      | Multilingual | 8192          |
| Fietje-2              | 2.7B      | English, Dutch continued | 2048          |
| Phi-3.5-mini-instruct | 3.8B      | Multilingual | 128000        |

## Perplexity and Bits per Word Evaluation

The model's performance was evaluated using two metrics. Token-level Perplexity (PPL) measures how well the model predicts the next token in a sequence. Since it normalizes the log-likelihood by the number of tokens, it is sensitive to tokenizer efficiency. Bits per Word (BPW) normalizes the log-likelihood of a sequence by words rather than tokens, making it more comparable across models with different tokenization strategies. For Dutch text, native Dutch gpt2 models use about 1.4 tokens per word, whereas Dutch+English Bor-1B uses about 1.5 tokens per word. Multilingual models such as SmolLM2, Llama, Phi, and Fietje use between 1.86 and 2.38 tokens per word on Dutch text. Consequently, direct comparisons based solely on PPL are less straightforward when tokenization differs significantly across models. Bits per Word (BPW) helps mitigate these discrepancies by focusing on per-word log-likelihood rather than per-token.

The evaluation used three datasets:
- **MC4 NL**: Dutch content from MC4
- **CulturaX NL**: High-quality Dutch cultural and educational content
- **Fineweb Edu**: English educational/technical content

### Token-level Perplexity

| dataset           |   gpt-neo-125M-dutch |   gpt2-medium-dutch |   gpt2-large-dutch |   Bor-1B |   Llama-3.2-1B |   gpt-neo-1.3B-dutch |   SmolLM2-1.7B |   Fietje-2 |   Phi-3.5-mini-instruct |
|:------------------|---------------------:|--------------------:|-------------------:|---------:|---------------:|---------------------:|---------------:|-----------:|------------------------:|
| MC4 (Dutch)       |                23.19 |               16.61 |              18.26 |    15.15 |          13    |                17.93 |          13.11 |       4.62 |                   10.73 |
| CulturaX (Dutch)  |                29.83 |               21.18 |              23.44 |    16.09 |          13.8  |                22.89 |          13.99 |       4.66 |                   11.47 |
| Fineweb (English) |                28.51 |               17.5  |              18.88 |    14.62 |          10.85 |                21.27 |           7.77 |      12.91 |                    6.92 |

### Bits per Word

| dataset           |   gpt-neo-125M-dutch |   gpt2-medium-dutch |   gpt2-large-dutch |   Bor-1B |   Llama-3.2-1B |   gpt-neo-1.3B-dutch |   SmolLM2-1.7B |   Fietje-2 |   Phi-3.5-mini-instruct |
|:------------------|---------------------:|--------------------:|-------------------:|---------:|---------------:|---------------------:|---------------:|-----------:|------------------------:|
| MC4 (Dutch)       |                 5.88 |                5.22 |               5.38 |     5.67 |           6.63 |                 5.34 |           8.52 |       4.85 |                    6.71 |
| CulturaX (Dutch)  |                 6.77 |                6.04 |               6.23 |     5.96 |           6.91 |                 6.2  |           8.84 |       4.95 |                    6.98 |
| Fineweb (English) |                 9.21 |                7.84 |               8.04 |     5.56 |           4.24 |                 8.4  |           3.67 |       4.66 |                    3.96 |

### Tokens per Word

| dataset           |   gpt-neo-125M-dutch |   gpt2-medium-dutch |   gpt2-large-dutch |   Bor-1B |   Llama-3.2-1B |   gpt-neo-1.3B-dutch |   SmolLM2-1.7B |   Fietje-2 |   Phi-3.5-mini-instruct |
|:------------------|---------------------:|--------------------:|-------------------:|---------:|---------------:|---------------------:|---------------:|-----------:|------------------------:|
| MC4 (Dutch)       |                 1.4  |                1.4  |               1.4  |     1.5  |           1.86 |                 1.4  |           2.35 |       2.31 |                    2.03 |
| CulturaX (Dutch)  |                 1.44 |                1.44 |               1.44 |     1.55 |           1.89 |                 1.44 |           2.38 |       2.33 |                    2.07 |
| Fineweb (English) |                 1.92 |                1.92 |               1.92 |     1.49 |           1.28 |                 1.92 |           1.32 |       1.31 |                    1.48 |

The violin plots show the distribution of Perplexity and Bits per Word metrics across models and datasets:

![Perplexity Distribution Across Models](perplexity_distribution.png)

Key observations for Bor-1B:
- Achieves competitive Bits per Word (~5.8) on Dutch text
- Maintains consistent performance across both Dutch and English content (CulturaX NL vs Fineweb Edu)

Full evaluation methodology: [evaluation script](https://github.com/yhavinga/bor-llm/blob/master/src/evaluate_entropy_metrics.py)

## Intended Use
- Rapid experimentation with Mistral architecture for Dutch/English tasks
- Single-task fine-tuning (e.g., text simplification, translation) where context contains all necessary information
- Research and development of Dutch language capabilities
- Lightweight applications requiring basic Dutch/English understanding

## Data Considerations and Limitations
- Tiny model size by current standards (~1.19B parameters and 0.2T tokens trained on) limits complex reasoning and generation capabilities
- Not suitable for high-quality chat responses from simple prompts
- Best performance achieved when full context/information is provided in the prompt
- Performance may vary between Dutch and English
- Domain-specific performance depends on training data distribution
- Architectural choice of maintaining 32 layers results in suboptimal inference speed for the parameter count
- As parts of the training data (CulturaX, MC4) are derived from CommonCrawl, the model may have been exposed to personal or sensitive information despite cleaning efforts

## License

### Model License
This model and its weights are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### Dataset Licenses
The training data includes content from multiple sources with different licenses:

- **CulturaX**: Inherits licenses from:
  - [mC4 license](https://huggingface.co/datasets/allenai/c4#license)
  - [OSCAR license](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information)
- **MC4-NL-cleaned**: ODC-BY license, subject to Common Crawl terms of use
- Additional datasets: See respective repository licenses

## Citations

If you use this model, please cite both the model and relevant datasets:

```bibtex
@article{2019t5,
    author        = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and
                     Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and
                     Peter J. Liu},
    title         = {Exploring the Limits of Transfer Learning with a Unified 
                     Text-to-Text Transformer},
    journal       = {arXiv e-prints},
    year          = {2019},
    archivePrefix = {arXiv},
    eprint        = {1910.10683}
}

@article{husain2019codesearchnet,
    title        = {{CodeSearchNet} Challenge: Evaluating the State of Semantic Code Search},
    author       = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and
                    Allamanis, Miltiadis and Brockschmidt, Marc},
    journal      = {arXiv preprint arXiv:1909.09436},
    year         = {2019}
}

@misc{Zare_Chavoshi_2023,
    title        = {EasyDeL, an Open-Source Library for Enhanced Training of JAX/Flax Models},
    url          = {https://github.com/erfanzar/EasyDeL},
    journal      = {EasyDeL Easy and Fast DeepLearning with JAX},
    publisher    = {Erfan Zare Chavoshi},
    author       = {Zare Chavoshi, Erfan},
    year         = {2023}
}

@misc{GoodWiki,
    title        = {GoodWiki Dataset},
    author       = {Choi, Euirim},
    howpublished = {\url{https://www.github.com/euirim/goodwiki}},
    month        = {September},
    year         = {2023}
}

@inproceedings{nguyen-etal-2024-culturax,
    title     = "{C}ultura{X}: A Cleaned, Enormous, and Multilingual Dataset for Large 
                 Language Models in 167 Languages",
    author    = "Nguyen, Thuat and others",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational 
                 Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    year      = "2024",
    pages     = "4226--4237"
}


@inproceedings{Huang2024OpenCoderTO,
    title        = {OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models},
    author       = {Huang, Siming and Cheng, Tianhao and Liu, Jason Klein and
                    Hao, Jiaran and Song, Liuyihan and Xu, Yang and Yang, J. and
                    Liu, J. H. and Zhang, Chenchen and Chai, Linzheng and
                    Yuan, Ruifeng and Zhang, Zhaoxiang and Fu, Jie and Liu, Qian and
                    Zhang, Ge and Wang, Zili and Qi, Yuan and Xu, Yinghui and
                    Chu, Wei},
    year         = {2024},
    url          = {https://arxiv.org/pdf/2411.04905}
}
```

## Acknowledgements

This project would not have been possible without compute generously provided by Google through the [TPU Research Cloud](https://sites.research.google/trc/). Training was implemented using [EasyDeL](https://erfanzar.github.io/EasyDeL/), an Apache v2 licensed open-source framework for efficient training of language models on TPU/GPU with JAX/Flax.

Created by [Yeb Havinga](https://www.linkedin.com/in/yeb-havinga-86530825/)
