#!/bin/bash
# Automated Phase 1 execution pipeline
# Run this after baseline comparisons (seed 1-3) are complete

set -e  # Exit on error

echo "==================================="
echo "Phase 1 Automated Pipeline"
echo "==================================="
echo ""

BASE_DIR="/data01/nakata/master_thesis/persona2"
cd "$BASE_DIR"

PERSONA_ID="episode-184019_A"
PROMPTS_FILE="persona-opt/episode-184019_A/eval_prompts.json"
SEED=1

# Create logs directory
mkdir -p logs/phase1

# ============================================
# Step 1: Check prerequisites
# ============================================
echo "[Step 1] Checking prerequisites..."

if [ ! -f "reports/experiments/results/baseline_comparison_seed1.json" ]; then
    echo "ERROR: Seed 1 results not found!"
    exit 1
fi

if [ ! -f "reports/experiments/results/baseline_comparison_seed2.json" ]; then
    echo "ERROR: Seed 2 results not found!"
    exit 1
fi

if [ ! -f "reports/experiments/results/baseline_comparison_seed3.json" ]; then
    echo "ERROR: Seed 3 results not found!"
    exit 1
fi

echo "✓ All baseline comparison results found"
echo ""

# ============================================
# Step 2: Aggregate multi-seed results
# ============================================
echo "[Step 2] Aggregating multi-seed results..."
python -B scripts/aggregate_multiseed_results.py

if [ $? -eq 0 ]; then
    echo "✓ Multi-seed aggregation complete"
else
    echo "ERROR: Multi-seed aggregation failed!"
    exit 1
fi
echo ""

# ============================================
# Step 3: Cross-layer evaluation
# ============================================
echo "[Step 3] Running cross-layer evaluation..."
echo "This will take 3-5 hours. Running in background..."

nohup python -B scripts/run_cross_layer_evaluation.py \
  --persona-id "$PERSONA_ID" \
  --prompts-file "$PROMPTS_FILE" \
  --num-prompts 20 \
  --seed $SEED \
  --methods Base Prompt_Persona MeanDiff PCA Random_Search Grid_Search Proposed \
  --layers 20 21 22 23 24 \
  > logs/phase1/cross_layer_eval.log 2>&1 &

CROSS_LAYER_PID=$!
echo "Cross-layer evaluation started (PID: $CROSS_LAYER_PID)"
echo "Monitor with: tail -f logs/phase1/cross_layer_eval.log"
echo ""

# Wait for cross-layer to complete
echo "Waiting for cross-layer evaluation to complete..."
wait $CROSS_LAYER_PID

if [ $? -eq 0 ]; then
    echo "✓ Cross-layer evaluation complete"
else
    echo "ERROR: Cross-layer evaluation failed!"
    exit 1
fi
echo ""

# ============================================
# Step 4: Generate layer heatmap
# ============================================
echo "[Step 4] Generating layer heatmap..."

# Find the most recent cross-layer results file
CROSS_LAYER_RESULT=$(ls -t reports/experiments/results/cross_layer_eval_*.json | head -1)

if [ -z "$CROSS_LAYER_RESULT" ]; then
    echo "ERROR: No cross-layer results found!"
    exit 1
fi

python -B scripts/generate_layer_heatmap.py --results-file "$CROSS_LAYER_RESULT"

if [ $? -eq 0 ]; then
    echo "✓ Layer heatmap generated"
else
    echo "ERROR: Heatmap generation failed!"
    exit 1
fi
echo ""

# ============================================
# Step 5: Ablation study
# ============================================
echo "[Step 5] Running ablation study..."
echo "This will take 1-2 hours. Running in background..."

nohup python -B scripts/run_ablation_study.py \
  --persona-id "$PERSONA_ID" \
  --prompts-file "$PROMPTS_FILE" \
  --num-prompts 10 \
  --seed $SEED \
  --layer 22 \
  > logs/phase1/ablation_study.log 2>&1 &

ABLATION_PID=$!
echo "Ablation study started (PID: $ABLATION_PID)"
echo "Monitor with: tail -f logs/phase1/ablation_study.log"
echo ""

# Wait for ablation to complete
echo "Waiting for ablation study to complete..."
wait $ABLATION_PID

if [ $? -eq 0 ]; then
    echo "✓ Ablation study complete"
else
    echo "ERROR: Ablation study failed!"
    exit 1
fi
echo ""

# ============================================
# Step 6: Generate figures
# ============================================
echo "[Step 6] Generating Phase 1 figures..."

# Find ablation results
ABLATION_RESULT=$(ls -t reports/experiments/results/ablation_study_*.json | head -1)

if [ -z "$ABLATION_RESULT" ]; then
    echo "ERROR: No ablation results found!"
    exit 1
fi

# Generate ablation bar chart
python -B scripts/generate_phase1_figures.py \
  --ablation-results "$ABLATION_RESULT" \
  --generate-ablation

if [ $? -ne 0 ]; then
    echo "ERROR: Ablation bar chart generation failed!"
    exit 1
fi

# Generate seed variation plot
python -B scripts/generate_phase1_figures.py \
  --generate-seed-variation

if [ $? -ne 0 ]; then
    echo "ERROR: Seed variation plot generation failed!"
    exit 1
fi

echo "✓ All figures generated"
echo ""

# ============================================
# Step 7: Generate final report
# ============================================
echo "[Step 7] Generating Phase 1 final report..."

python -B scripts/generate_phase1_report.py

if [ $? -eq 0 ]; then
    echo "✓ Phase 1 final report generated"
else
    echo "ERROR: Report generation failed!"
    exit 1
fi
echo ""

# ============================================
# Summary
# ============================================
echo "==================================="
echo "Phase 1 Pipeline Complete!"
echo "==================================="
echo ""

echo "Generated artifacts:"
echo ""
echo "Tables:"
ls -lh reports/experiments/tables/*.md | awk '{print "  - " $9}'
echo ""
echo "Figures:"
ls -lh reports/experiments/figures/*.png | awk '{print "  - " $9}'
echo ""
echo "Report:"
echo "  - reports/experiments/phase1_final_report.md"
echo ""

echo "View report:"
echo "  cat reports/experiments/phase1_final_report.md"
echo ""

echo "Next steps: Phase 2 experiments"
echo "  - TruthfulQA evaluation"
echo "  - MMLU evaluation"
echo "  - Multi-judge validation"
echo ""
