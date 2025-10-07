#!/bin/bash
set -euo pipefail

# Configuration
IMAGE_NAME="ubuntu-rocm-hf"
IMAGE_TAG="6.3.0"
#DOCKER_FILE="Dockerfile-ubuntu-rocm-mi300x-nocondaenv"
DOCKER_FILE="Dockerfile-ubuntu-rocm-mi300-pytorch-training-based"

GPU_TYPE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --mi300x) GPU_TYPE="mi300x"; shift ;;
        --7900xtx) GPU_TYPE="7900xtx"; shift ;;
        --all) GPU_TYPE="all"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Validate input
if [ -z "$GPU_TYPE" ]; then
    echo "Usage: $0 --mi300x|--7900xtx|--all"
    exit 1
fi

# Set GPU architecture based on selection
case $GPU_TYPE in
    "mi300x")
        GPU_ARCH="gfx942"
        IMAGE_TAG="${IMAGE_TAG}-mi300x"
        ;;
    "7900xtx")
        GPU_ARCH="gfx1100"
        IMAGE_TAG="${IMAGE_TAG}-7900xtx"
        ;;
    "all")
        GPU_ARCH="gfx942:gfx1100"
        IMAGE_TAG="${IMAGE_TAG}-unified"
        ;;
esac

echo "Building Docker image for ${GPU_TYPE}..."
docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${DOCKER_FILE}" \
    --build-arg ROCM_VERSION="${IMAGE_TAG%%-*}" \
    --build-arg GPU_ARCH="${GPU_ARCH}" \
    .

echo -e "\nTo create a Singularity SIF file on aacjump, run the following commands:"
echo "# Save docker image as tar file"
echo "docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz"
echo "# Copy to aacjump"
echo "scp ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz aacjump:~/"
echo "# SSH to aacjump and build singularity image"
echo "ssh aacjump 'gunzip -c ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz | singularity build ${IMAGE_NAME}_${IMAGE_TAG}.sif docker-archive:/dev/stdin'"
echo "# Clean up local tar file"
echo "rm ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz" 