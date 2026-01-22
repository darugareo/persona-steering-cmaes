#!/bin/bash

# Phase 1: Base, Random, Few-shot evaluation
# 21 personas × 3 conditions × ~12 turns = ~756 evaluations
# Estimated time: 2-3 hours

env CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_all_conditions.py \
    --output_dir results/condition_comparison \
    --conditions base random fewshot \
    2>&1 | tee phase1_evaluation.log
