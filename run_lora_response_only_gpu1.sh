#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python scripts/train_lora_response_only.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir lora_models_response_only \
    --num_epochs 3 \
    --batch_size 4 \
    --personas episode-179307_A episode-184019_A episode-19493_A \
                episode-223194_B episode-24275_A episode-36796_A \
                episode-36796_B episode-74475_A episode-84804_A \
                episode-98323_A
