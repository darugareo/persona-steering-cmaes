#!/bin/bash
# Run 2-Trait Steering experiment on GPU 0 (2 personas)

echo "================================================================================"
echo "2-Trait Steering Experiment - GPU 0 (2 personas)"
echo "================================================================================"
echo ""
echo "Target personas:"
echo "  1. episode-184019_A"
echo "  2. episode-118328_B"
echo ""
echo "Settings:"
echo "  - GPU: 0"
echo "  - Traits: R2, R4 only (2D optimization)"
echo "  - Generations: 20"
echo "  - Population: 8"
echo ""
echo "================================================================================"
echo ""

# Create log directory
mkdir -p logs/two_trait_optimization

PERSONAS=(
    "episode-184019_A"
    "episode-118328_B"
)

for PERSONA in "${PERSONAS[@]}"; do
    echo ""
    echo "================================================================================"
    echo "Starting optimization: $PERSONA"
    echo "================================================================================"

    CUDA_VISIBLE_DEVICES=0 python scripts/two_trait_optimizer.py \
        --persona_id "$PERSONA" \
        --gpu_id 0 \
        --max_generations 20 \
        --popsize 8 \
        2>&1 | tee "logs/two_trait_optimization/${PERSONA}_gpu0.log"

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
echo "GPU 0 optimizations complete!"
echo "================================================================================"
