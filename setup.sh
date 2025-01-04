#!/bin/bash

# Exit on error
set -e

# Python version
PYTHON_CMD="python3.10"

# Print usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: ./setup.sh <environment-type>"
    echo "Available environment types:"
    echo "  torch-cuda  - PyTorch with CUDA support"
    echo "  torch-rocm  - PyTorch with ROCm support"
    echo "  jax-cuda   - JAX with CUDA support"
    echo "  jax-tpu    - JAX with TPU support"
    exit 1
fi

ENV_TYPE=$1

# Map environment types to directory names
case $ENV_TYPE in
    "torch-cuda")
        VENV_PATH="venvtorchcuda"
        ;;
    "torch-rocm")
        VENV_PATH="venvtorchrocm"
        ;;
    "jax-cuda")
        VENV_PATH="venvjaxcuda"
        ;;
    "jax-tpu")
        VENV_PATH="venvjaxtpu"
        ;;
    *)
        echo "Error: Unknown environment type: $ENV_TYPE"
        echo "Available environment types: torch-cuda, torch-rocm, jax-cuda, jax-tpu"
        exit 1
        ;;
esac

# Create and activate virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new virtual environment in $VENV_PATH"
    $PYTHON_CMD -m venv "$VENV_PATH"
else
    echo "Using existing virtual environment in $VENV_PATH"
fi

# Initialize and update git submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install packages based on environment type
case $ENV_TYPE in
    "torch-cuda")
        pip install -r requirements_torch_cuda.txt
        # Install flash-attention separately with no build isolation
        pip install flash-attn --no-build-isolation
        # Install Axolotl with flash-attention
        pip install --no-build-isolation "axolotl[flash-attn,deepspeed]"
        
        # Fetch DeepSpeed configs if directory doesn't exist
        if [ ! -d "deepspeed_configs" ]; then
            echo "Fetching Axolotl DeepSpeed configs..."
            axolotl fetch deepspeed_configs
        fi
        ;;
    "torch-rocm")
        # Install required dependencies first
        sudo apt update
        sudo apt install -y python3-dev
        pip install -r requirements_torch_rocm.txt
        ;;
    "jax-cuda")
        pip install -r requirements_jax_cuda.txt
        ;;
    "jax-tpu")
        pip install -r requirements_jax_tpu.txt
        ;;
    *)
        echo "Error: Unknown environment type: $ENV_TYPE"
        echo "Available environment types: torch-cuda, torch-rocm, jax-cuda, jax-tpu"
        exit 1
        ;;
esac

# Install git-lfs if not already installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing Git LFS..."
    sudo apt-get update && sudo apt-get install -y git-lfs
    git lfs install
else
    echo "Git LFS already installed"
fi

# Check if logged into Hugging Face
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to Hugging Face:"
    huggingface-cli login
else
    echo "Already logged into Hugging Face"
fi

# Check if logged into Wandb
if ! wandb login &> /dev/null; then
    echo "Please login to Wandb:"
    wandb login
else
    echo "Already logged into Wandb"
fi

echo "Environment setup complete. Activated: $ENV_TYPE"
echo "To activate this environment later, run:"
echo "source ${VENV_PATH}/bin/activate"

# Perform environment-specific tests
case $ENV_TYPE in
    "torch-cuda"|"torch-rocm")
        echo "Testing PyTorch installation..."
        $PYTHON_CMD -c 'import torch' 2>/dev/null && echo "PyTorch import: Success" || echo "PyTorch import: Failure"
        $PYTHON_CMD -c 'import torch; print(f"GPU available: {torch.cuda.is_available()}")' 
        ;;
    "jax-cuda"|"jax-tpu")
        echo "Testing JAX installation..."
        $PYTHON_CMD -c 'import jax' 2>/dev/null && echo "JAX import: Success" || echo "JAX import: Failure"
        ;;
esac 