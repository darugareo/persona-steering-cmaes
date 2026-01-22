#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u scripts/evaluate_lora.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora_dir lora_models_response_only \
    --output_dir results/lora_evaluation_response_only
