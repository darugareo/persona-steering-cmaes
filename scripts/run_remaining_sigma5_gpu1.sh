#!/bin/bash
# GPU 1: 残り11ペルソナの後半5個を実行

OUTPUT_DIR="optimization_results_sigma5"
SIGMA=5.0
ALPHA=2.0
GPU=1

PERSONAS=(
    "episode-51953_A"
    "episode-74475_A"
    "episode-84804_A"
    "episode-98323_A"
    "episode-98947_A"
)

echo "============================================================"
echo "GPU $GPU: Remaining σ=5.0 Optimization (5 personas)"
echo "============================================================"
echo "Output: $OUTPUT_DIR"
echo ""

for i in "${!PERSONAS[@]}"; do
    PERSONA=${PERSONAS[$i]}
    NUM=$((i+1))
    TOTAL=${#PERSONAS[@]}

    echo ""
    echo "[$NUM/$TOTAL] Optimizing: $PERSONA (GPU $GPU)"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=$GPU python scripts/optimize_single_persona_sigma5.py \
        --persona_id "$PERSONA" \
        --output_dir "$OUTPUT_DIR" \
        --sigma "$SIGMA" \
        --alpha "$ALPHA" \
        --max_iterations 50 \
        --population_size 8

    if [ $? -eq 0 ]; then
        echo "✅ $PERSONA complete"
    else
        echo "❌ $PERSONA failed"
    fi
done

echo ""
echo "============================================================"
echo "GPU $GPU: All 5 personas completed"
echo "============================================================"
