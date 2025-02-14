#!/bin/bash
#SBATCH --job-name=fsdp_train          # Job name (optional)
#SBATCH --nodes=2                      # Number of nodes
#SBATCH --ntasks=2                     # Total tasks (1 per node when using torchrun)
#SBATCH --gres=gpu:4                   # GPUs per node (4 GPUs on each node)
#SBATCH --cpus-per-task=96            # CPU cores per task (all cores on each node)
#SBATCH --partition=192C4G1H_MI300A_RHEL8   # Partition name (for MI300A nodes)
#SBATCH --time=500:00:00               # Runtime limit (500 hours)
#SBATCH --output=outputs/out/fsdp_training_out.txt   # Standard output log
#SBATCH --error=outputs/err/fsdp_training_err.txt    # Standard error log

# Load any necessary modules (e.g., Singularity) and prepare environment
module load singularity 2>/dev/null   # Load Singularity if not already in PATH
# Activate conda environment if needed (inside or outside container)
# e.g., source ~/anaconda3/etc/profile.d/conda.sh && conda activate myenv

# Enable strict error handling for robust execution
set -euo pipefail
# (Optional) Print commands and host for debugging:
# set -x
echo "Running on host $(hostname) with $SLURM_JOB_NUM_NODES nodes"

# Set distributed training variables
export MASTER_ADDR=$(srun --ntasks=1 hostname --ip-address | awk '{print $1}')
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export WORLD_SIZE=$(( GPUS_PER_NODE * SLURM_JOB_NUM_NODES ))

echo "MASTER_ADDR=$MASTER_ADDR, WORLD_SIZE=$WORLD_SIZE"

# Run the FSDP training across 2 nodes and 8 GPUs total using torchrun
echo "Starting multi-node FSDP training via torchrun..."
srun singularity exec -B .:/workdir \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env GPUS_PER_NODE=$GPUS_PER_NODE \
    --env WORLD_SIZE=$WORLD_SIZE \
    --env SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES \
    --env SLURM_JOB_ID=$SLURM_JOB_ID \
    --rocm pytorch_training_v25_2.sif \
    bash -c -eux 'source activate base && \
    conda activate py_3.10 && \
    cd /workdir/bor-llm && \
    torchrun \
        --nnodes=${SLURM_JOB_NUM_NODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        --rdzv_id=${SLURM_JOB_ID} \
        src/finetune/finetune_bor_trl_multinode.py'

# Note: torchrun will spawn 4 processes per node (one per GPU), for a total of 8 processes.
# Each process will run src/finetune/finetune_bor_trl_4gpus.py.
# The rendezvous backend 'c10d' ensures all processes connect.
# The job will end when the training script completes on all ranks.
