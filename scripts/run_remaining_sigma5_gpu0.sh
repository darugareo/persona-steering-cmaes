#!/bin/bash
# GPU 0: 残り11ペルソナの前半6個を実行

OUTPUT_DIR="optimization_results_sigma5"
SIGMA=5.0
ALPHA=2.0
GPU=0

PERSONAS=(
    "episode-239427_A"
    "episode-24275_A"
    "episode-36796_A"
    "episode-36796_B"
    "episode-37624_A"
    "episode-38144_A"
)

echo "============================================================"
echo "GPU $GPU: Remaining σ=5.0 Optimization (6 personas)"
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
echo "GPU $GPU: All 6 personas completed"
echo "============================================================"
