# Bor-1B-v1 - Dutch/English Language Model

## Model Description

<img src="bor-de-wolf.jpg" align="left" width="190" style="margin-right: 15px; margin-bottom: 10px;">

Bor-1B-v1 is a pre-trained multilingual language model based on the Mistral architecture, specifically designed for Dutch and English language processing. The model was pre-trained in two phases with different context lengths and dataset compositions.

The model's name pays homage to "Bor de Wolf" from the beloved Dutch children's series De Fabeltjeskrant (1968-1989). While the original Bor was famous for his signature "Huuuuu!" howl echoing through the forest, this Bor's howls are more subtle - processing Dutch and English text without disturbing the neighbors!

<div style="clear: both;"></div>

## Architecture
The Mistral architecture was chosen for its broad library support and efficient implementation. The Bor-v1-1B model maintains Mistral's core structural elements while scaling down dimensions for a more compact design:

- **Model Scale**: ~1.1B parameters vs Mistral's 7B, primarily due to reduced width:
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

![BOR Pretraining Dataset Composition](bor-v1-pretrain-datasets.png)

### Phase 1 (2048 context length)
Total tokens: 37B tokens × 3 epochs = 112B total
- Primary Dutch content (~80%): [MC4 Dutch Cleaned](https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned), Wikipedia,
 [GeminiPhiDutch](https://huggingface.co/datasets/Kalamazooter/GeminiPhiDutch) and additional Dutch datasets
- English content (~14%) from curated sources, including [goodwiki](https://huggingface.co/datasets/euirim/goodwiki)
- Python code (~6%) for programming capabilities: Python subsets from [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) and [github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean/viewer/Python-all)

### Phase 2 (4096 context length)
Total tokens: 37B tokens × 2.4 epochs = 87B total
- Balanced composition: Dutch (55%), English (35%), and specialized content (~10%)
- Dutch content: [MC4 Dutch Cleaned](https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned), Wikipedia,
 [GeminiPhiDutch](https://huggingface.co/datasets/Kalamazooter/GeminiPhiDutch), [CulturaX Dutch](https://huggingface.co/datasets/uonlp/CulturaX/viewer/nl)
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
- Learning rate: 2.18e-4 → 5e-5 (linear decay)
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

![Training Loss for Bor-v1](bor-v1-1b-training-loss.png)

- Final training loss stabilized around 2.1-2.2 after initial optimization
- Phase 2 (4096 context) shows higher loss, most likely due to dataset composition:
  - More diverse content mix (35% English vs 14% in Phase 1)
  - Addition of technical content (mathematics, specialized code)
- Multiple warmup cycles visible but maintained consistent loss levels

## Intended Use
- Rapid experimentation with Mistral architecture for Dutch/English tasks
- Single-task fine-tuning (e.g., text simplification, translation) where context contains all necessary information
- Research and development of Dutch language capabilities
- Lightweight applications requiring basic Dutch/English understanding

## Limitations
- Extremely small model size by current standards (~1.1B parameters) limits complex reasoning and generation capabilities
- Not suitable for high-quality chat responses from simple prompts
- Performance may vary between Dutch and English
- Domain-specific performance depends on training data distribution
- Model behaviors inherit biases present in training data
- Best performance achieved when full context/information is provided in the prompt
- Architectural choice of maintaining 32 layers results in suboptimal inference speed for the parameter count - throughput may be lower than other ~1B parameter models with more balanced depth/width ratios

## License
Apache 2.0

This model and its weights are licensed under the Apache License 2.0. See LICENSE file for details.

## Acknowledgements

This project would not have been possible without compute generously provided by Google through the [TPU Research Cloud](https://sites.research.google/trc/). Training was implemented using [EasyDeL](https://erfanzar.github.io/EasyDeL/), an open-source framework for efficient training of language models on TPU/GPU with JAX/Flax.

Created by [Yeb Havinga](https://www.linkedin.com/in/yeb-havinga-86530825/)