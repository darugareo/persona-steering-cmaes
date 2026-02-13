#!/bin/bash
# Generate steering vectors for layer 15 and layer 25

export CUDA_VISIBLE_DEVICES=1
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
LAYERS="15,25"

for TRAIT in R1 R2 R3 R4 R5; do
    echo "========================================"
    echo "Generating vectors for trait: $TRAIT"
    echo "========================================"

    python scripts/run_build_svd_vectors.py \
        --positive data/prompts/extracted/${TRAIT}_positive.json \
        --negative data/prompts/extracted/${TRAIT}_negative.json \
        --layers $LAYERS \
        --model $MODEL \
        --save_dir data/steering_vectors_v2/$TRAIT \
        --dtype bfloat16

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate vectors for $TRAIT"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All vectors generated successfully!"
echo "========================================"
