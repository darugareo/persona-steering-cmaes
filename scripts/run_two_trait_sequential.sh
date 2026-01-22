#!/bin/bash
# Run 2-Trait Steering experiment SEQUENTIALLY on GPU 1 only
# To avoid GPU conflicts with rumesh on GPU 0

echo "================================================================================"
echo "2-Trait Steering Experiment - Sequential Mode (GPU 1 only)"
echo "================================================================================"
echo ""
echo "Target personas:"
echo "  1. episode-184019_A"
echo "  2. episode-118328_B"
echo "  3. episode-239427_A"
echo "  4. episode-225888_A"
echo ""
echo "Settings:"
echo "  - GPU: 1 only (avoiding GPU 0 conflict with rumesh)"
echo "  - Traits: R2, R4 only (2D optimization)"
echo "  - Generations: 20"
echo "  - Population: 8"
echo "  - Mode: Sequential (one at a time)"
echo ""
echo "================================================================================"
echo ""

# Create log directory
mkdir -p logs/two_trait_optimization

PERSONAS=(
    "episode-184019_A"
    "episode-118328_B"
    "episode-239427_A"
    "episode-225888_A"
)

for PERSONA in "${PERSONAS[@]}"; do
    echo ""
    echo "================================================================================"
    echo "Starting optimization: $PERSONA"
    echo "================================================================================"

    CUDA_VISIBLE_DEVICES=1 python scripts/two_trait_optimizer.py \
        --persona_id "$PERSONA" \
        --gpu_id 0 \
        --max_generations 20 \
        --popsize 8 \
        2>&1 | tee "logs/two_trait_optimization/${PERSONA}.log"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $PERSONA completed successfully"
    else
        echo "❌ $PERSONA failed with exit code $EXIT_CODE"
    fi

    echo ""
done

echo ""
echo "================================================================================"
echo "All optimizations complete!"
echo "================================================================================"
