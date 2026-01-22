#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train_lora_response_only.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir lora_models_response_only \
    --num_epochs 3 \
    --batch_size 4 \
    --personas episode-118328_B episode-128744_B episode-134226_A \
                episode-136981_B episode-137872_B episode-140544_B \
                episode-14330_A episode-145896_A episode-158821_B \
                episode-16276_B episode-175246_A
