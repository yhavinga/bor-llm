# BOR LLM Project

A multilingual Dutch/English language model using the Mistral architecture with a custom pre-trained model.

## Features

- Custom pre-trained multilingual model for Dutch and English
- Implemented using MistralForCausalLM architecture
- Dutch-LLaMA tokenizer integration
- Multilingual model supporting Dutch and English
- Built on Mistral 1.1B architecture
- Multi-platform support (CUDA/ROCm/JAX/TPU)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/bor-llm.git
cd bor-llm
```

2. Set up the environment based on your hardware:
```
./setup.sh <environment-type>
```

Available environment types:
- `torch-cuda`: PyTorch with CUDA support
- `torch-rocm`: PyTorch with ROCm support  
- `jax-cuda`: JAX with CUDA support
- `jax-tpu`: JAX with TPU support

## Usage

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
- CUDA/ROCm compatible GPU or TPU
- Git LFS
- Hugging Face account
- Weights & Biases account

## License

MIT License

Copyright (c) 2024 [Your Organization Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.