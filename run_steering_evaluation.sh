#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u scripts/evaluate_steering_sigma5.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --sigma5_dir optimization_results_sigma5 \
    --output_dir results/steering_evaluation_sigma5
