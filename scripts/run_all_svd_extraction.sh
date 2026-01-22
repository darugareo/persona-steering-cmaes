#!/bin/bash

# SVD Vector Generation for All Traits
# Run this to generate all 25 steering vectors (5 traits × 5 layers)

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
LAYERS="20,21,22,23,24"
BASE_DIR="data"

echo "=========================================="
echo "SVD Steering Vector Generation"
echo "=========================================="
echo "Model: $MODEL"
echo "Layers: $LAYERS"
echo ""

for TRAIT in R1 R2 R3 R4 R5; do
    echo "----------------------------------------"
    echo "Processing trait: $TRAIT"
    echo "----------------------------------------"

    python scripts/run_build_svd_vectors.py \
        --positive ${BASE_DIR}/prompts/extracted/${TRAIT}_positive.json \
        --negative ${BASE_DIR}/prompts/extracted/${TRAIT}_negative.json \
        --layers ${LAYERS} \
        --model ${MODEL} \
        --save_dir ${BASE_DIR}/steering_vectors_v2/${TRAIT}/ \
        --dtype bfloat16

    if [ $? -eq 0 ]; then
        echo "✅ $TRAIT completed successfully"
    else
        echo "❌ $TRAIT failed"
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "✅ All SVD vectors generated successfully!"
echo "=========================================="
echo ""
echo "Output location:"
echo "  ${BASE_DIR}/steering_vectors_v2/"
echo ""
echo "Generated files:"
ls -lh ${BASE_DIR}/steering_vectors_v2/*/layer*.pt
