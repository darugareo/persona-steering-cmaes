#!/bin/bash
#
# Simple evaluation script for episode-184019_A
# Runs all 6 evaluation tasks sequentially
#

PERSONA_ID="episode-184019_A"
PROMPTS_FILE="persona-opt/episode-184019_A/eval_prompts.json"

echo "============================================================"
echo "Persona Steering Evaluation Pipeline"
echo "============================================================"
echo "Persona: $PERSONA_ID"
echo "Prompts: $PROMPTS_FILE"
echo "============================================================"
echo ""

# Note: Full implementation requires PersonaAwareEvaluator class
# which wraps evaluate_with_persona_judge for batch operations.
# Current codebase uses direct function calls.

echo "[INFO] Evaluation modules created:"
echo "  - persona_opt/evaluation/train_test.py"
echo "  - persona_opt/evaluation/cross_layer.py"
echo "  - persona_opt/evaluation/alpha_sensitivity.py"
echo "  - persona_opt/evaluation/multi_turn.py"
echo "  - persona_opt/evaluation/multi_judge.py"
echo "  - persona_opt/evaluation/human_eval.py"
echo ""

echo "[INFO] CLI scripts created:"
echo "  - scripts/run_train_test.py"
echo "  - scripts/run_cross_layer.py"
echo "  - scripts/run_alpha_sensitivity.py"
echo "  - scripts/run_multi_turn.py"
echo "  - scripts/run_multi_judge.py"
echo "  - scripts/run_human_eval_template.py"
echo ""

echo "[INFO] Documentation:"
echo "  - docs/EVALUATION_GUIDE.md"
echo ""

echo "[NOTE] To run evaluations, you need to:"
echo "1. Implement PersonaAwareEvaluator wrapper class that:"
echo "   - Wraps evaluate_with_persona_judge() for batch operations"
echo "   - Provides batch_evaluate() method"
echo "   - Handles conversation history for multi-turn"
echo ""
echo "2. Or modify evaluation modules to use evaluate_with_persona_judge() directly"
echo ""

echo "============================================================"
echo "Files Ready"
echo "============================================================"
ls -lh persona-opt/episode-184019_A/
echo ""
echo "Optimization results:"
cat persona-opt/episode-184019_A/best_weights.json
echo ""
echo "============================================================"
