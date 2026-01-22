#!/bin/bash
# 全ペルソナに対してAdaptive Trait最適化を実行

RECOMMENDATIONS_FILE="trait_recommendations_threshold1.5.json"
OUTPUT_DIR="optimization_results_adaptive"
GPU_ID=0

# Extract persona IDs from recommendations
PERSONAS=$(python3 -c "
import json
with open('$RECOMMENDATIONS_FILE') as f:
    data = json.load(f)
    personas = [p for p, rec in data['recommendations'].items() if rec['selected_traits']]
    print(' '.join(personas))
")

echo "========================================================================"
echo "Adaptive Trait Optimization for All Personas"
echo "========================================================================"
echo "Recommendations file: $RECOMMENDATIONS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "Total personas with selected traits: $(echo $PERSONAS | wc -w)"
echo "========================================================================"
echo ""

# Run optimization for each persona
for PERSONA_ID in $PERSONAS; do
    echo "--------------------------------------------------------------------"
    echo "Processing: $PERSONA_ID"
    echo "--------------------------------------------------------------------"

    python scripts/run_adaptive_trait_optimization.py \
        --persona_id "$PERSONA_ID" \
        --recommendations "$RECOMMENDATIONS_FILE" \
        --gpu_id $GPU_ID \
        --max_generations 20 \
        --popsize 8 \
        --warm_start \
        --output_dir "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✅ $PERSONA_ID completed"
    else
        echo "❌ $PERSONA_ID failed"
    fi

    echo ""
done

echo "========================================================================"
echo "All optimizations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================================"
