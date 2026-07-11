#!/usr/bin/env bash
# Build and push the self-contained submission image to a PUBLIC Docker Hub repo.
# Usage: scripts/build_and_push.sh <dockerhub-user>/<repo>:<tag>
set -euo pipefail

IMAGE="${1:?usage: build_and_push.sh <user>/<repo>:<tag>}"

if [ ! -d "model" ]; then
  echo "ERROR: ./model not found. Download (fixed-hash) weights and/or run offline quantize first." >&2
  exit 1
fi

docker build -f docker/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
echo "Pushed $IMAGE (ensure the Docker Hub repo is PUBLIC before submitting)."
