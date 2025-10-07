#!/bin/bash
#SBATCH --job-name=bor-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --partition=192C4G1H_MI300A_RHEL8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Add these debugging flags
set -x  # Print commands and their arguments as they are executed
set -e  # Exit immediately if a command exits with a non-zero status

# Print some debug info
pwd
ls -la
echo "Working directory: $PWD"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Load any necessary modules if required
# module load ...

# Set environment variables
export PYTORCH_TUNABLEOP_ENABLED=1
export HIP_FORCE_DEV_KERNARG=1
export NCCL_MIN_NCHANNELS=112
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_VISIBLE_DEVICES=0,1,2,3
# export HF_TOKEN=<your-token-here>

# Execute the training
singularity exec \
    -B .:/workdir \
    -B /shared/midgard/home/buisman_hnu/.cache/huggingface:/root/.cache/huggingface \
    /shared/apps/rhel8/cscs-2024-11-12/UWV/bitsandbytes/docker-image-rocm624.sif \
    bash -c -eux '$WITH_CONDA ; conda activate pytorch ; \
    pip install transformers~=4.47.0 \
                accelerate~=1.2.0 \
                sentencepiece~=0.2.0 \
                datasets \
                huggingface-hub>=0.24.0 \
                safetensors>=0.4.1 \
                peft \
                trl \
                numpy \
                pandas \
                matplotlib \
                seaborn \
                wandb \
                tensorboard \
                wheel \
                pytest \
                evaluate \
                tabulate \
                plotext ; \
    cd /workdir/bor-llm ; \
    ls -la ; \
    accelerate launch \
        src/finetune/finetune_bor_trl.py \
        --model_name_or_path "yhavinga/Bor-1B" \
        --dataset_path "/workdir/dataset-bor/openhermes_leesplank_20250205_061457" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --lora_r 16 \
        --logging_steps 1 \
        --report_to "stdout" \
        --log_level "info"'
