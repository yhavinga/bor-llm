#!/bin/bash
#SBATCH --job-name=fsdp_train
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=192
#SBATCH --partition=192C4G1H_MI300A_RHEL8
#SBATCH --time=500:00:00
#SBATCH --output=outputs/check-node-network-out.txt
#SBATCH --error=outputs/check-node-network-err.txt

module load singularity 2>/dev/null

set -euo pipefail

echo "Running on host $(hostname) with $SLURM_JOB_NUM_NODES nodes"

# Set distributed training variables
export MASTER_ADDR=$(srun --ntasks=1 hostname --ip-address | awk '{print $1}')
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export WORLD_SIZE=$(( GPUS_PER_NODE * SLURM_JOB_NUM_NODES ))

echo "MASTER_ADDR=$MASTER_ADDR, WORLD_SIZE=$WORLD_SIZE"

# Run the new python driver script that detects working interfaces and launches training
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
    python src/check_node_network.py'
