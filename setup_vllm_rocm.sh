#!/bin/bash
set -e

# 0. Clone vLLM repository
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 7a8987dac5f0ed0c798a73e8b4ec8f5e640bc63a

# 1. Setup environment
python3.10 -m venv venvrocm
source venvrocm/bin/activate
pip install --upgrade pip

# 2. Install PyTorch for ROCm
pip install torch==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2

# 3. Install build dependencies
pip install ninja~=1.11.1.3 cmake~=3.31.4 wheel pybind11~=2.13.6 "setuptools>=61" setuptools-scm

# 4. Install custom Triton build
pip uninstall -y triton
git clone https://github.com/OpenAI/triton.git
cd triton
git checkout e192dba
cd python
pip install .
cd ../..

# 5. Install other dependencies
pip install "numpy<2" numba~=0.61.0 scipy~=1.15.1 huggingface-hub[cli]~=0.28.1
pip install -r requirements-rocm.txt

# 6. Set environment variables
export ROCM_HOME=/opt/rocm-6.2.3
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$ROCM_HOME
export HIP_PLATFORM=amd
export PYTORCH_HIP_SUPPORT=1
export PYTORCH_ROCM_ARCH="gfx1100"  # Adjust for your GPU architecture

# 7. Install vLLM
python3 setup.py develop