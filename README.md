# BOR LLM Project

[![Tests](https://github.com/yhavinga/bor-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/yhavinga/bor-llm/actions/workflows/tests.yml)

Bor-1B is a 1.19B parameter language model optimized for Dutch and English, built on the Mistral architecture. This repository contains the training, evaluation, and deployment code for the model.

## Features

- 1.19B parameter model with strong Dutch and English capabilities
- Built on Mistral architecture with optimized dimensions
- Custom Dutch-LLaMA tokenizer for efficient bilingual processing
- Comprehensive training pipeline with multi-platform support
- Evaluation tools and deployment scripts
- Supports CUDA, ROCm, JAX, and TPU backends

## Installation

1. Clone the repository:
```git clone https://github.com/yourusername/bor-llm.git
cd bor-llm
```

2. Set up the environment based on your hardware:
```
./setup.sh <environment-type>
```

Available environment types:
- `torch-cuda`: PyTorch with CUDA support
- `torch-rocm`: PyTorch with ROCm support
- `torch-cpu`: PyTorch with CPU-only support
- `jax-cuda`: JAX with CUDA support
- `jax-tpu`: JAX with TPU support

## Usage

### Environment Setup

For testing purposes, you can skip the Hugging Face and Weights & Biases authentication:
```bash
TESTING=1 ./setup.sh <environment-type>
```

### Finetuning

1. Configure your finetuning parameters in `configs/finetune/bor_finetune.yml`

2. Preprocess the dataset:
```
CUDA_VISIBLE_DEVICES="0" axolotl preprocess configs/finetune/bor_finetune.yml
```
3. Launch multi-GPU training:
```
accelerate launch -m axolotl.cli.train configs/finetune/bor_finetune.yml
```

### Model Evaluation

After finetuning, you can evaluate the model in two ways:

1. Create an Ollama model:
```
python src/create_ollama_model_from_checkpoint.py
```

2. Run the vibe check script for direct evaluation:
```
python src/bor_vibe_check.py
```

## Project Structure

```
.
├── configs/             # Configuration files
│   └── finetune/        # Finetuning configs
├── src/                 # Source code
├── deepspeed_configs/   # DeepSpeed configuration files
├── requirements_*.txt   # Environment-specific requirements
└── setup.sh             # Environment setup script
```

## Technical Requirements

- Python 3.10+
- CUDA/ROCm compatible GPU, TPU, or CPU-only setup
- Git LFS
- Hugging Face account (optional in testing mode)
- Weights & Biases account (optional in testing mode)

## License
Apache 2.0

This model and its weights are licensed under the Apache License 2.0. See LICENSE file for details.
