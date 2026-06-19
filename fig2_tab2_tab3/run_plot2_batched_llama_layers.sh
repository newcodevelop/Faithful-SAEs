#!/usr/bin/env bash

set -u

cd "$(dirname "$0")"
mkdir -p runs

for layer in 4 8 12 16 20 24 28 30; do
    echo "Starting layer ${layer}..."
    CUDA_VISIBLE_DEVICES=1 nohup python -u plot2_batched_llama_layered.py \
        --layer "$layer" \
        &> "./runs/plot2_batched_llama_layer${layer}.out"
    echo "Completed layer ${layer}."
done
