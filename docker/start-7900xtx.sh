#!/bin/bash
set -euo pipefail

IMAGE_NAME="ubuntu-rocm-hf"
IMAGE_TAG="6.3.2-7900xtx"
CONTAINER_NAME="rocm-dev-7900xtx"

# Check if container exists and is stopped
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "Reattaching to existing container..."
    docker start -ai "$CONTAINER_NAME"
else
    echo "Creating new container..."
    docker run --rm -it \
        --name "$CONTAINER_NAME" \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v $(pwd):/workdir \
        -w /workdir \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        bash
fi