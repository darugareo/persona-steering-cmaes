#!/bin/bash
# Run 2-Trait Steering experiment on GPU 1 (2 personas)

echo "================================================================================"
echo "2-Trait Steering Experiment - GPU 1 (2 personas)"
echo "================================================================================"
echo ""
echo "Target personas:"
echo "  1. episode-239427_A"
echo "  2. episode-225888_A"
echo ""
echo "Settings:"
echo "  - GPU: 1"
echo "  - Traits: R2, R4 only (2D optimization)"
echo "  - Generations: 20"
echo "  - Population: 8"
echo ""
echo "================================================================================"
echo ""

# Create log directory
mkdir -p logs/two_trait_optimization

PERSONAS=(
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
        2>&1 | tee "logs/two_trait_optimization/${PERSONA}_gpu1.log"

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
echo "GPU 1 optimizations complete!"
echo "================================================================================"
